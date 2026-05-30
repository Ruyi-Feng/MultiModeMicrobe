
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
                 property_attn: bool = False,
                 llm_property: bool = True):
        super(MicrobeCLIP, self).__init__()

        self.collective = collective
        self.property_attn = property_attn
        self.llm_property = llm_property

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
        # numerical encoder
        if not self.llm_property:
            self.property_encoder = property_encoder
            self._property_proj = nn.Linear(property_encoder.embedding_dim, cross_hidden_size, bias=False)
            for module in self.property_encoder.embedding_layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(self._property_proj.weight, gain=1.0)
            return
        # descriptor encoder
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
        self.multihead_attn = nn.MultiheadAttention(cross_hidden_size, 4, 0.1, batch_first=True)
        self._property_proj = nn.Linear(hidden_size, cross_hidden_size, bias=False)
        # 使用Xavier初始化投影层，有助于训练稳定性
        nn.init.xavier_uniform_(self.multihead_attn.in_proj_weight, gain=1.0)
        nn.init.xavier_uniform_(self._property_proj.weight, gain=1.0)

    def _init_aa_net(self, aa_encoder, trainable, aa_representation_dim, cross_hidden_size):
        if not self.collective:
            if aa_encoder is None:
                raise ValueError("aa_encoder must be provided when collective is False")

            if aa_encoder.name == "microbe_protein_repr":
                in_dim = getattr(aa_encoder, "output_dim", aa_encoder.embed_dim)
                self._aa_proj = nn.Linear(in_dim, cross_hidden_size, bias=False)
                # 使用Xavier初始化投影层
                nn.init.xavier_uniform_(self._aa_proj.weight, gain=1.0)
                # 用于贴在前面提取表征的向量，使用更小的初始化范围
                self.protein_cls_token = nn.Parameter(torch.randn(1, 1, aa_representation_dim) * 0.02)
            if aa_encoder.name in ["gumbal_softmax", "attention_convergence"]:
                in_dim = getattr(aa_encoder, "output_dim", aa_encoder.embed_dim)
                self._aa_proj = nn.Linear(in_dim, cross_hidden_size, bias=False)
                # 使用Xavier初始化投影层
                nn.init.xavier_uniform_(self._aa_proj.weight, gain=1.0)
            self.aa_encoder = aa_encoder
            grad_adjustment(aa_encoder, trainable['aa_encoder'])

        if self.collective:
            self._aa_proj = nn.Linear(aa_representation_dim, cross_hidden_size, bias=False)
            # 使用Xavier初始化投影层
            nn.init.xavier_uniform_(self._aa_proj.weight, gain=1.0)


    def _forward_llm_property(self, property_seq, property_cls_token_index=None):
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
        return property_embedding, property_cls_token.squeeze(1)

    def _forward_numerical_property(self, property_seq):
        # property_seq: [batch_size, 4]
        property_embedding = self.property_encoder(property_seq)
        property_embedding = self._property_proj(property_embedding)
        return property_embedding

    def _forward_aa(self, aa_rep, padding_mask=None, return_weights=False):
        if self.collective:
            if aa_rep.ndim != 2:
                raise ValueError("aa_rep must be 2D (batch, dim) when collective is True, please use collective features")
            aa_embedding = self._aa_proj(aa_rep)  # B, H
            return aa_embedding, None
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
                aa_embedding, aa_weights = self.aa_encoder(aa_rep, padding_mask=padding_mask, return_weights=True)
            else:
                aa_embedding = self.aa_encoder(aa_rep)
            aa_embedding = self._aa_proj(aa_embedding)
            if return_weights:
                return aa_embedding, aa_weights
            else:
                return aa_embedding, None

    def forward(self, aa_seq, property_seq, property_cls_token_index=None, return_hidden_states=False, padding_mask=None, return_weights=False):

        aa_embedding, aa_weights = self._forward_aa(aa_seq, padding_mask=padding_mask, return_weights=return_weights)
        aa_embedding = aa_embedding / aa_embedding.norm(dim=1, keepdim=True)

        if self.llm_property:
            _, property_cls_token = self._forward_llm_property(property_seq, property_cls_token_index)
            property_cls_token = property_cls_token / property_cls_token.norm(dim=1, keepdim=True)
        else:
            property_cls_token = self._forward_numerical_property(property_seq)

        # 约束logit_scale在合理范围内，防止数值不稳定
        logit_scale = torch.clamp(self.logit_scale, -np.log(100), np.log(100)).exp()
        logits_aa = logit_scale * aa_embedding @ property_cls_token.t()
        logits_property = logits_aa.t()

        if return_hidden_states:
            pred = {
                "aa_representation": aa_embedding,
                "property_representation": property_cls_token,
                "logits_aa": logits_aa,
                "logits_property": logits_property,
                "aa_weights": aa_weights
            }
            return pred
        else:
            pred = {
                "logits_aa": logits_aa,
                "logits_property": logits_property,
                "aa_weights": aa_weights
            }
            return pred


MISSING_LABEL = -1.0  # 必须与 property_encoder.MISSING_VALUE 保持一致


class E2EPrediction(nn.Module):
    def __init__(self,
                 aa_encoder,
                 hidden_size: int = 128,
                 property_dim: int = 3,
                 per_property_head: bool = True,
                 pred_dropout: float = 0.1,
                 ):
        super(E2EPrediction, self).__init__()
        self.hidden_size = hidden_size
        self.property_dim = property_dim
        self.per_property_head = per_property_head
        self._init_aa_net(aa_encoder)
        self._init_pred_head(hidden_size, property_dim, pred_dropout)

    def _init_aa_net(self, aa_encoder):
        # 新版 AttentionConvergence 在多 query / mean-max 拼接下，输出维度
        # 不再等于 embed_dim，必须读 output_dim。
        in_dim = getattr(aa_encoder, "output_dim", aa_encoder.embed_dim)
        self._aa_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        nn.init.xavier_uniform_(self._aa_proj[0].weight, gain=1.0)
        nn.init.zeros_(self._aa_proj[0].bias)
        self.aa_encoder = aa_encoder
        if getattr(aa_encoder, "name", None) == "microbe_protein_repr":
            self.protein_cls_token = nn.Parameter(
                torch.randn(1, 1, aa_encoder.embed_dim) * 0.02
            )

    def _init_pred_head(self, hidden_size, property_dim, dropout):
        def make_head(out_dim):
            mid = max(hidden_size // 2, 16)
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid, out_dim),
            )

        if self.per_property_head:
            # 每个 property 一个独立 head，避免 4 个性状共享同一组特征做最后一层映射
            self.pred_layer = nn.ModuleList([make_head(1) for _ in range(property_dim)])
        else:
            self.pred_layer = make_head(property_dim)

    def forward(self, aa_seq, padding_mask=None):
        aa_weights = None
        if getattr(self.aa_encoder, "name", None) == "microbe_protein_repr":
            cls_token = self.protein_cls_token.expand(aa_seq.size(0), -1, -1)
            aa_seq = torch.cat([cls_token, aa_seq], dim=1)
            if padding_mask is not None:
                cls_mask = torch.ones(
                    aa_seq.size(0), 1,
                    dtype=padding_mask.dtype,
                    device=padding_mask.device,
                )
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        if getattr(self.aa_encoder, "name", None) == "attention_convergence" and padding_mask is not None:
            aa_embedding, aa_weights = self.aa_encoder(aa_seq, padding_mask=padding_mask, return_weights=True)
        else:
            aa_embedding = self.aa_encoder(aa_seq)

        aa_embedding = self._aa_proj(aa_embedding)

        if self.per_property_head:
            pred = torch.cat([head(aa_embedding) for head in self.pred_layer], dim=-1)
        else:
            pred = self.pred_layer(aa_embedding)

        return {
            "pred": pred,
            "aa_weights": aa_weights,
        }

    def get_loss(self,
                 logits,
                 labels,
                 attn_weights=None,
                 attn_sparsity_lambda: float = 0.0,
                 missing_value: float = MISSING_LABEL):
        """Masked MSE：缺失值（== missing_value）不进 loss，避免 -1 哨兵污染回归目标。

        可选：对 attention weights 加熵正则（lambda * mean entropy），鼓励聚合稀疏地选蛋白。
        """
        if labels.size(-1) > logits.size(-1):
            labels = labels[:, :logits.size(-1)]

        valid = (labels != missing_value).float()
        diff_sq = (logits - labels) ** 2
        denom = valid.sum().clamp(min=1.0)
        loss = (diff_sq * valid).sum() / denom

        if attn_sparsity_lambda > 0 and attn_weights is not None:
            # aa_weights 可能是 (B, S) 或 (B, num_queries, S)
            w = attn_weights.clamp(min=1e-9)
            entropy = -(w * w.log()).sum(dim=-1)  # (B,) 或 (B, num_queries)
            loss = loss + attn_sparsity_lambda * entropy.mean()

        return loss


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
    """聚合 (B, S, D) -> (B, output_dim)。

    旧版只有一个 tanh+v 的标量 score，softmax over S，每个菌株只挑出一组"重要蛋白"
    并把 3000+ 蛋白压成单一加权向量；对温度/pH/盐/氧 4 个互相独立的性状来说瓶颈极窄。

    新版默认开启：
      - K 个可学习 query token + multi-head attention pooling（Set Transformer 的 PMA）
        让模型对不同性状分配不同的"重要蛋白集合"。
      - 可选 protein-protein self-attention 层（让蛋白之间能交互）。
      - 可选 mean / max 统计与 attention pooling 拼接（三视角池化更鲁棒）。

    输出维度 = num_queries * embed_dim  (+ 2 * embed_dim 如果 concat_mean_max)。

    向后兼容：num_queries=1 & num_heads=1 & self_attn_layers=0 & not concat_mean_max
    时退化为旧的 tanh+v 行为，output_dim=embed_dim。
    """

    def __init__(self,
                 embed_dim,
                 hidden_dim=None,
                 num_queries: int = 4,
                 num_heads: int = 4,
                 self_attn_layers: int = 1,
                 concat_mean_max: bool = True,
                 dropout: float = 0.1,
                 *args, **kwargs):
        super().__init__()
        self.name = "attention_convergence"
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.self_attn_layers = self_attn_layers
        self.concat_mean_max = concat_mean_max

        self._legacy = (num_queries == 1 and num_heads == 1
                        and self_attn_layers == 0 and not concat_mean_max)

        if self._legacy:
            if hidden_dim is None:
                hidden_dim = embed_dim
            self.linear = nn.Linear(embed_dim, hidden_dim)
            self.v = nn.Parameter(torch.randn(hidden_dim) * 0.02, requires_grad=True)
            self.output_dim = embed_dim
            return

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # 可选的 protein-to-protein self-attention（pre-norm transformer block）
        if self_attn_layers > 0:
            self.self_attn_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(self_attn_layers)
            ])
            self.self_attns = nn.ModuleList([
                nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
                for _ in range(self_attn_layers)
            ])
            self.ffn_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(self_attn_layers)
            ])
            self.ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(self_attn_layers)
            ])

        # PMA：K 个可学习 query 跨多头注意力，每个 query 出一个汇聚向量
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.pool_q_norm = nn.LayerNorm(embed_dim)
        self.pool_k_norm = nn.LayerNorm(embed_dim)
        self.pma = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.pool_out_norm = nn.LayerNorm(embed_dim)

        out_dim = num_queries * embed_dim
        if concat_mean_max:
            out_dim += 2 * embed_dim
        self.output_dim = out_dim

    def _legacy_forward(self, aa_repr, padding_mask=None, return_weights=False):
        x = self.linear(aa_repr)
        e = torch.matmul(F.tanh(x), self.v)
        all_padding = None
        if padding_mask is not None:
            e = e.masked_fill(~padding_mask, float('-inf'))
            valid_lengths = padding_mask.sum(dim=1)
            all_padding = valid_lengths == 0
        weights = F.softmax(e, dim=1)
        if padding_mask is not None and all_padding is not None and all_padding.any():
            uniform_weights = torch.ones_like(weights) / weights.size(1)
            weights = torch.where(all_padding.unsqueeze(1), uniform_weights, weights)
        out = (aa_repr * weights.unsqueeze(-1)).sum(dim=1)
        if return_weights:
            return out, weights
        return out

    def forward(self, aa_repr, padding_mask=None, return_weights=False):
        """
        Args:
            aa_repr: (B, S, embed_dim)
            padding_mask: (B, S) bool, True 表示有效，False 表示 padding
        Returns:
            (B, output_dim) [, weights]
            weights: 单 query 时 (B, S)；多 query 时 (B, num_queries, S)
        """
        if self._legacy:
            return self._legacy_forward(aa_repr, padding_mask, return_weights)

        B = aa_repr.size(0)
        # nn.MultiheadAttention 的 key_padding_mask 语义是 True=ignore
        key_padding_mask = None
        all_pad = None
        if padding_mask is not None:
            key_padding_mask = ~padding_mask
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                # 防 NaN：全 padding 的样本把第 0 位强制设为有效
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False

        # 1) protein-protein self-attention（让蛋白之间交互）
        x = aa_repr
        for i in range(self.self_attn_layers):
            x_norm = self.self_attn_norms[i](x)
            attn_out, _ = self.self_attns[i](
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = x + attn_out
            x_norm = self.ffn_norms[i](x)
            x = x + self.ffns[i](x_norm)

        # 2) multi-query attention pooling (PMA)
        Q = self.queries.expand(B, -1, -1)
        Q_n = self.pool_q_norm(Q)
        K_n = self.pool_k_norm(x)
        pooled, weights = self.pma(
            Q_n, K_n, K_n,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        # pooled: (B, num_queries, D); weights: (B, num_queries, S)
        pooled = self.pool_out_norm(pooled)
        out = pooled.flatten(1)  # (B, num_queries * D)

        # 3) 可选 mean / max 与 attention pooling 拼接（三视角池化）
        if self.concat_mean_max:
            if padding_mask is not None:
                mask_f = padding_mask.unsqueeze(-1).float()
                denom = mask_f.sum(dim=1).clamp(min=1.0)
                mean_pool = (x * mask_f).sum(dim=1) / denom
                masked_for_max = x.masked_fill(~padding_mask.unsqueeze(-1), float('-inf'))
                max_pool = masked_for_max.max(dim=1).values
                # 全 padding 的样本 max 会是 -inf，置 0
                max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)
            else:
                mean_pool = x.mean(dim=1)
                max_pool = x.max(dim=1).values
            out = torch.cat([out, mean_pool, max_pool], dim=1)

        if return_weights:
            return out, weights
        return out


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