import torch
import torch.nn as nn
import esm


class ESM2:
    def __init__(self, model_name):
        if model_name == 'esm2_t6_8M_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.repr_layers = [6]
        if model_name == 'esm2_t12_35M_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.repr_layers = [12]
        if model_name == 'esm2_t30_150M_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            self.repr_layers = [30]
        if model_name == 'esm2_t33_650M_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layers = [33]
        if model_name == 'esm2_t36_3B_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            self.repr_layers = [36]
        if model_name == 'esm2_t48_15B_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            self.repr_layers = [48]
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, seqs):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(seqs)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.model(batch_tokens, repr_layers=self.repr_layers, return_contacts=True)
        token_representations = results["representations"][self.repr_layers[-1]]
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        return sequence_representations
