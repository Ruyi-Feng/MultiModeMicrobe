
from model.backbone import Microbe
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import esm

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


def test_load_qwen_decoder():
    model_path = "Qwen/Qwen-1_8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    inputs = tokenizer(
    "菌株属性有xxx",
      return_tensors='pt')

    inputs = inputs.to(model.device)
    pred = model(**inputs, output_hidden_states=True, output_attentions=False, return_dict=True)
    last_hidden_states = pred.hidden_states[-1]


def test_esm2_as_aaencoder():
    aa_encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    property_encoder = nn.Embedding(10, 512, padding_idx=0)
    trainable = {'aa_encoder': True, 'property_encoder': True, 'llm_decoder': True}

    aa_seq = torch.randint(0, 9, (4, 10))   # batch = 4, seq_len = 10
    property_seq = torch.randint(0, 9, (4, 10))
    model = Microbe(aa_encoder, 33, property_encoder, trainable, cross_hidden_size=512)
    pred = model(aa_seq, property_seq)

    assert pred.shape == (4, 1)