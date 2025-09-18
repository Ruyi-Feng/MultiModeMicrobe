
import torch.nn as nn


class Microbe(nn.Module):
    def __init__(self, aa_encoder, property_encoder, llm_decoder, trainable: dict,
                 cross_hidden_size: int = 128):
        super(Microbe, self).__init__()

        self._grad_adjustment(aa_encoder, trainable['aa_encoder'])
        self._grad_adjustment(property_encoder, trainable['property_encoder'])
        self._grad_adjustment(llm_decoder, trainable['llm_decoder'])

        self.aa_encoder = aa_encoder
        self.property_encoder = property_encoder
        self.llm_decoder = llm_decoder
        self._aa_proj = nn.Linear(aa_encoder.embedding_dim, cross_hidden_size, bias=False)
        self._property_proj = nn.Linear(property_encoder.embedding_dim, cross_hidden_size, bias=False)
        self._llm_proj = nn.Linear(cross_hidden_size, llm_decoder.in_features, bias=False)
        self._alignment = CrossAttention(cross_hidden_size, num_heads=8, dropout=0.1)

    def _grad_adjustment(self, model, required_grad):
        if required_grad == True:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False

        return

    def forward(self, aa_seq, property_seq):
        aa_embedding = self.aa_encoder(aa_seq)   # B, S, H_a
        aa_embedding = self._aa_proj(aa_embedding)  # B, S, H
        property_embedding = self.property_encoder(property_seq)
        property_embedding = self._property_proj(property_embedding)

        crossed_embedding = self._alignment(aa_embedding, property_embedding, property_embedding)[0]
        cross_embedding = self._llm_proj(crossed_embedding)

        llm_decoding = self.llm_decoder(cross_embedding)
        return llm_decoding


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
