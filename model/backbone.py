
import torch
import torch.nn as nn

class Microbe(nn.Module):
    def __init__(self, aa_encoder, aa_layer_num, property_encoder, trainable: dict,
                 cross_hidden_size: int = 128, output_hidden_states=False):
        super(Microbe, self).__init__()

        grad_adjustment(aa_encoder, trainable['aa_encoder'])
        grad_adjustment(property_encoder, trainable['property_encoder'])

        self.aa_encoder = aa_encoder
        self.aa_layer_num = aa_layer_num
        self.property_encoder = property_encoder
        self._aa_proj = nn.Linear(aa_encoder.embed_dim, cross_hidden_size, bias=False)
        self._property_proj = nn.Linear(property_encoder.embedding_dim, cross_hidden_size, bias=False)
        self._alignment = CrossAttentionFusion(cross_hidden_size, num_heads=8, dropout=0.1)
        self.output_hidden_states = output_hidden_states

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cross_hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)  # 初始化

        self.classifier = nn.Linear(cross_hidden_size, 1)

    def forward(self, aa_seq, property_seq):
        aa_embedding = self.aa_encoder(aa_seq, repr_layers=[self.aa_layer_num], return_contacts=True)   # B, S, H_a
        aa_embedding = aa_embedding['representations'][self.aa_layer_num]
        aa_embedding = self._aa_proj(aa_embedding)  # B, S, H

        property_embedding = self.property_encoder(property_seq)
        property_embedding = self._property_proj(property_embedding)

        B = property_embedding.size(0)

        # 将可学习的 [CLS] token 扩展到 batch 维度，并拼接到 property 序列开头
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1,1,H) -> (B,1,H)
        property_embedding = torch.cat([cls_tokens, property_embedding], dim=1)  # (B, S_p+1, H)
        crossed_embedding = self._alignment(property_embedding, aa_embedding)

        cls_output = crossed_embedding[:, 0, :]  # (B, H)
        pred = self.classifier(cls_output)
        if self.output_hidden_states:
            return pred, cls_output  # 分类结果，特征
        else:
            return pred  # 直接返回分类结果


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