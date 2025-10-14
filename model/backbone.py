
import torch
import torch.nn as nn
import numpy as np



class MicrobeCLIP(nn.Module):
    """
    这个版本是直接输入esm处理好的aa的representation
    相当于先不考虑esm的finetune
    """
    def __init__(self, aa_layer_num, property_encoder, trainable: dict,
                 cross_hidden_size: int = 128,
                 aa_representation_dim: int = 128):
        super(Microbe, self).__init__()

        grad_adjustment(property_encoder, trainable['property_encoder'])

        self.aa_layer_num = aa_layer_num
        self.property_encoder = property_encoder
        self._aa_proj = nn.Linear(aa_representation_dim, cross_hidden_size, bias=False)
        self._property_proj = nn.Linear(property_encoder.config.hidden_size, cross_hidden_size, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(cross_hidden_size, 1),
            nn.Sigmoid()
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, aa_rep, property_seq, aa_cls_token_index=0, property_cls_token_index=0, return_hidden_states=False):

        aa_embedding = self._aa_proj(aa_rep)  # B, S, H

        property_embedding = self.property_encoder(**property_seq, output_hidden_states=True, output_attentions=False, return_dict=True)
        property_embedding = property_embedding.hidden_states[-1].float()
        property_embedding = self._property_proj(property_embedding)   # 这里还是有seq len在的。所以还是要加cls token来做全局表征。
        property_cls_token = property_embedding[:, property_cls_token_index, :]
        property_embedding = property_embedding[:, 1:, :]

        aa_embedding = aa_embedding / aa_embedding.norm(dim=1, keepdim=True)
        property_cls_token = property_cls_token / property_cls_token.norm(dim=1, keepdim=True)


        logit_scale = self.logit_scale.exp()
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


class Microbe(nn.Module):
    def __init__(self, aa_encoder, aa_layer_num, property_encoder, trainable: dict,
                 cross_hidden_size: int = 128):
        super(Microbe, self).__init__()

        grad_adjustment(aa_encoder, trainable['aa_encoder'])
        grad_adjustment(property_encoder, trainable['property_encoder'])

        self.aa_encoder = aa_encoder
        self.aa_layer_num = aa_layer_num
        self.property_encoder = property_encoder
        self._aa_proj = nn.Linear(aa_encoder.embed_dim, cross_hidden_size, bias=False)
        self._property_proj = nn.Linear(property_encoder.config.hidden_size, cross_hidden_size, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(cross_hidden_size, 1),
            nn.Sigmoid()
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, aa_seq, property_seq, aa_cls_token_index=0, property_cls_token_index=0, return_hidden_states=False):

        aa_embedding = self.aa_encoder(aa_seq, repr_layers=[self.aa_layer_num], return_contacts=True)   # B, S, H_a
        aa_embedding = aa_embedding['representations'][self.aa_layer_num]
        aa_embedding = self._aa_proj(aa_embedding)  # B, S, H
        aa_cls_token = aa_embedding[:, aa_cls_token_index, :]
        aa_embedding = aa_embedding[:, 1:, :]

        property_embedding = self.property_encoder(**property_seq, output_hidden_states=True, output_attentions=False, return_dict=True)
        property_embedding = property_embedding.hidden_states[-1].float()
        property_embedding = self._property_proj(property_embedding)   # 这里还是有seq len在的。所以还是要加cls token来做全局表征。
        property_cls_token = property_embedding[:, property_cls_token_index, :]
        property_embedding = property_embedding[:, 1:, :]

        aa_cls_token = aa_cls_token / aa_cls_token.norm(dim=1, keepdim=True)
        property_cls_token = property_cls_token / property_cls_token.norm(dim=1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits_aa = logit_scale * aa_cls_token @ property_cls_token.t()
        logits_property = logits_aa.t()


        if return_hidden_states:
            pred = {
                "aa_representation": aa_cls_token,
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


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, text_emb, aa_emb):
        # text_emb: (B, L_t, D)
        # image_emb: (B, L_v, D)
        attn_out, _ = self.cross_attn(
            query=text_emb,    # Query
            key=aa_emb,     # Key
            value=aa_emb    # Value
        )
        out = self.norm(text_emb + attn_out)
        out = self.norm(out + self.ffn(out))
        return out  # 对齐后的文本表示

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: (batch, tgt_len, embed_dim) —— Decoder 的输入
            key:   (batch, src_len, embed_dim) —— Encoder 的输出
            value: (batch, src_len, embed_dim) —— 通常和 key 相同
            key_padding_mask: (batch, src_len) —— 可选，mask 掉无效位置（如 padding）
        Returns:
            attn_output: (batch, tgt_len, embed_dim)
            attn_weights: (batch, tgt_len, src_len) —— 注意力权重（可选）
        """
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        return attn_output, attn_weights

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