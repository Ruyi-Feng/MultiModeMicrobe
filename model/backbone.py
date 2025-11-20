
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MicrobeCLIP(nn.Module):
    """
    这个版本是直接输入esm处理好的aa的representation
    相当于先不考虑esm的finetune
    默认是collective的
    individual的时候,输入的aa_encoder用来处理microbe的多个蛋白质序列获得菌株总体表征
    """
    def __init__(self,
                 property_encoder,
                 trainable: dict,
                 aa_encoder = None,
                 collective: bool = True,
                 cross_hidden_size: int = 128,
                 aa_representation_dim: int = 128,
                 property_attn: bool = False):
        super(MicrobeCLIP, self).__init__()

        self.collective = collective
        self.property_attn = property_attn

        self._init_aa_net(aa_encoder, trainable, aa_representation_dim, cross_hidden_size)
        self._init_property_net(property_encoder, trainable, cross_hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(cross_hidden_size, 1),
            nn.Sigmoid()
        )

        # logit_scale初始化为CLIP标准值，并添加约束防止过大
        # 使用clamp确保温度参数在合理范围内 [log(1/100), log(100)]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _init_property_net(self, property_encoder, trainable, cross_hidden_size):
        # 检测是否使用 LoRA：检查参数名中是否有 'lora_'
        is_lora = any('lora_' in name for name, _ in property_encoder.named_parameters())

        if trainable['property_encoder']:
            # 如果可训练
            if is_lora:
                pass
            else:
                grad_adjustment(property_encoder, True)
        else:
            grad_adjustment(property_encoder, False)

        self.property_encoder = property_encoder
        hidden_size = property_encoder.config.hidden_size
        self.multihead_attn = nn.MultiheadAttention(hidden_size, 4, 0.1, batch_first=True)
        self._property_proj = nn.Linear(hidden_size, cross_hidden_size, bias=False)
        # 使用Xavier初始化投影层，有助于训练稳定性
        nn.init.xavier_uniform_(self.multihead_attn.in_proj_weight, gain=1.0)
        nn.init.xavier_uniform_(self._property_proj.weight, gain=1.0)

    def _init_aa_net(self, aa_encoder, trainable, aa_representation_dim, cross_hidden_size):
        if not self.collective:
            if aa_encoder is None:
                raise ValueError("aa_encoder must be provided when collective is False")

            if aa_encoder.name == "microbe_protein_repr":
                self._aa_proj = nn.Linear(aa_encoder.embed_dim, cross_hidden_size, bias=False)
                # 使用Xavier初始化投影层
                nn.init.xavier_uniform_(self._aa_proj.weight, gain=1.0)
                # 用于贴在前面提取表征的向量，使用更小的初始化范围
                self.protein_cls_token = nn.Parameter(torch.randn(1, 1, aa_representation_dim) * 0.02)
            if aa_encoder.name in ["gumbal_softmax", "attention_convergence"]:
                self._aa_proj = nn.Linear(aa_encoder.embed_dim, cross_hidden_size, bias=False)
                # 使用Xavier初始化投影层
                nn.init.xavier_uniform_(self._aa_proj.weight, gain=1.0)
            self.aa_encoder = aa_encoder
            grad_adjustment(aa_encoder, trainable['aa_encoder'])

        if self.collective:
            self._aa_proj = nn.Linear(aa_representation_dim, cross_hidden_size, bias=False)
            # 使用Xavier初始化投影层
            nn.init.xavier_uniform_(self._aa_proj.weight, gain=1.0)


    def _forward_property(self, property_seq, property_cls_token_index=None):
        property_embedding = self.property_encoder(**property_seq, output_hidden_states=True, output_attentions=False, return_dict=True)
        property_embedding = property_embedding.hidden_states[-1].float()
        property_embedding = self._property_proj(property_embedding)   # 这里还是有seq len在的。所以还是要加cls token来做全局表征。
        if property_cls_token_index is None:
            property_cls_token = property_embedding[:, 0, :]
        else:
            batch_size = property_embedding.size(0)
            property_cls_token = property_embedding[torch.arange(batch_size), property_cls_token_index, :]
        property_embedding = property_embedding[:, 1:, :]

        if self.property_attn:
            property_cls_token, _ = self.multihead_attn(property_cls_token.unsqueeze(1), property_embedding,
                                                 property_embedding,
                                                 need_weights=False,
                                                 average_attn_weights=True)  # 如果需要分开每个头，这里false
        return property_embedding, property_cls_token

    def _forward_aa(self, aa_rep, padding_mask=None):
        if self.collective:
            if aa_rep.ndim != 2:
                raise ValueError("aa_rep must be 2D (batch, dim) when collective is True, please use collective features")
            aa_embedding = self._aa_proj(aa_rep)  # B, H
            return aa_embedding
        else:
            # 如果是cross_attention_fusion，需要加上cls token 再做attention
            if self.aa_encoder.name == "microbe_protein_repr":
                cls_token = self.protein_cls_token.expand(aa_rep.size(0), -1, -1)
                aa_rep = torch.concat([cls_token, aa_rep], dim=1)
                # 更新 padding_mask 以包含 cls_token
                if padding_mask is not None:
                    cls_mask = torch.ones(aa_rep.size(0), 1, dtype=padding_mask.dtype, device=padding_mask.device)
                    padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            if self.aa_encoder.name == "attention_convergence" and padding_mask is not None:
                aa_embedding = self.aa_encoder(aa_rep, padding_mask=padding_mask)
            else:
                aa_embedding = self.aa_encoder(aa_rep)
            aa_embedding = self._aa_proj(aa_embedding)
            return aa_embedding

    def forward(self, aa_seq, property_seq, property_cls_token_index=None, return_hidden_states=False, padding_mask=None):

        aa_embedding = self._forward_aa(aa_seq, padding_mask=padding_mask)
        _, property_cls_token = self._forward_property(property_seq, property_cls_token_index)

        aa_embedding = aa_embedding / aa_embedding.norm(dim=1, keepdim=True)
        property_cls_token = property_cls_token / property_cls_token.norm(dim=1, keepdim=True)

        # 约束logit_scale在合理范围内，防止数值不稳定
        logit_scale = torch.clamp(self.logit_scale, -np.log(100), np.log(100)).exp()
        logits_aa = logit_scale * aa_embedding @ property_cls_token.t()
        logits_property = logits_aa.t()

        if return_hidden_states:
            pred = {
                "aa_representation": aa_embedding,
                "property_representation": property_cls_token,
                "logits_aa": logits_aa,
                "logits_property": logits_property
            }
            return pred
        else:
            pred = {
                "logits_aa": logits_aa,
                "logits_property": logits_property
            }
            return pred


class GumbalSoftmax(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.name = "gumbal_softmax"
        self.temperature = torch.nn.Parameter(torch.ones(1) * 0.7)
        self.embed_dim = embed_dim
        # 将每个位置的特征转换为标量logit，用于生成序列级别的权重
        self.logit_proj = nn.Linear(embed_dim, 1)

    def forward(self, aa_repr):
        """
        Args:
            aa_repr: (B, S, D)
        Returns:
            x: (B, D)
        """
        # aa_repr = F.gumbel_softmax(aa_repr, tau=self.temperature, hard=True, dim=2)
        # x = aa_repr.sum(dim=1)
        # 为每个序列位置生成logit: (B, S, 1)
        logits = self.logit_proj(aa_repr).squeeze(-1)  # (B, S)
        # 使用Gumbel Softmax在序列维度上生成权重: (B, S)
        weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1)
        # 使用权重对序列特征进行加权求和: (B, D)
        x = (aa_repr * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return x


class AttentionConvergence(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, *args, **kwargs):
        super().__init__()
        self.name = "attention_convergence"
        self.embed_dim = embed_dim
        if hidden_dim is None:
            hidden_dim = embed_dim
        self.linear = nn.Linear(embed_dim, hidden_dim)
        # 使用较小的初始化值，与代码库中其他参数初始化保持一致
        self.v = nn.Parameter(torch.randn(hidden_dim) * 0.02, requires_grad=True)

    def forward(self, aa_repr, padding_mask=None):
        """
        Args:
            aa_repr: (B, S, embed_dim) 输入序列表示
            padding_mask: (B, S) 可选，True 表示有效位置，False 表示 padding
        Returns:
            x: (B, embed_dim) 聚合后的序列表示
        """
        x = self.linear(aa_repr)
        e = torch.matmul(F.tanh(x), self.v)  # (B, S, hidden_dim) @ (hidden_dim,) -> (B, S)

        all_padding = None
        if padding_mask is not None:
            e = e.masked_fill(~padding_mask, float('-inf'))
            valid_lengths = padding_mask.sum(dim=1)  # (B,)
            all_padding = valid_lengths == 0
        weights = F.softmax(e, dim=1)  # (B, S)
        if padding_mask is not None and all_padding is not None and all_padding.any():
            uniform_weights = torch.ones_like(weights) / weights.size(1)
            weights = torch.where(all_padding.unsqueeze(1), uniform_weights, weights)
        x = (aa_repr * weights.unsqueeze(-1)).sum(dim=1)  # (B, S, embed_dim) -> (B, embed_dim)
        return x


class MicrobeProteinRepr(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.name = "microbe_protein_repr"
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        # 为每一层创建独立的 LayerNorm，避免共享导致的问题
        self.attn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for i in range(self.num_layers):
            # Pre-norm architecture (更稳定)
            x_norm = self.attn_norms[i](x)
            attn_out, _ = self.self_attns[i](
                query=x_norm,
                key=x_norm,
                value=x_norm
            )
            x = x + attn_out  # 残差连接

            # FFN with pre-norm
            x_norm = self.ffn_norms[i](x)
            ffn_out = self.ffns[i](x_norm)
            x = x + ffn_out  # 残差连接
        repr = x[:, 0, :]
        return repr


class LLMDecoder(nn.Module):
    def __init__(self, head_module, llm_decoder, trainable: dict, cross_hidden_size: int = 128):
        super(LLMDecoder, self).__init__()

        grad_adjustment(llm_decoder, trainable['llm_decoder'])
        self.head_module = head_module
        self.llm_decoder = llm_decoder
        self._llm_proj = nn.Linear(cross_hidden_size, llm_decoder.config.hidden_size, bias=False)

    def forward(self, aa_seq, property_seq):
        crossed_embedding = self.head_module(aa_seq, property_seq)[0]

        cross_embedding = self._llm_proj(crossed_embedding)
        llm_decoding = self.llm_decoder(cross_embedding)
        pass


def grad_adjustment(model, required_grad):
    if required_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

    return