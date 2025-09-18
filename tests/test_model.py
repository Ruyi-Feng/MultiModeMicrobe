
from model.backbone import Microbe
import torch
import torch.nn as nn

def test_backbone():
    aa_encoder = nn.Embedding(10, 16, padding_idx=0)
    property_encoder = nn.Embedding(10, 16, padding_idx=0)
    llm_decoder = nn.Linear(32, 32)
    trainable = {'aa_encoder': True, 'property_encoder': True, 'llm_decoder': True}

    aa_seq = torch.randint(0, 9, (4, 10))   # batch = 4, seq_len = 10
    property_seq = torch.randint(0, 9, (4, 10))
    model = Microbe(aa_encoder, property_encoder, llm_decoder, trainable, cross_hidden_size=32)
    llm_decoding = model(aa_seq, property_seq)

    assert llm_decoding.shape == (4, 10, 32)

