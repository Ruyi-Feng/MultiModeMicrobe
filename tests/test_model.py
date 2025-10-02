
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
    aa_encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    aa_encoder = aa_encoder.to("cuda")
    trainable = {'aa_encoder': False, 'property_encoder': False, 'llm_decoder': True}

    model_path = "Qwen/Qwen-1_8B"
    property_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    property_tokenizer.pad_token = "<|endoftext|>"
    property_tokenizer.pad_token_id = property_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    property_encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    property_encoder = property_encoder.to("cuda")

    aa_seq = torch.randint(0, 9, (4, 10)).to("cuda")   # batch = 4, seq_len = 10
    descriptions = [
        "Domain: Bacteria, Phylum: Bacillota, Class: Bacilli, ",
        "Domain: Bacteria, Phylum: Proteobacteria, Class: Gammaproteobacteria, ",
        "Domain: Archaea, Phylum: Euryarchaeota, Class: Methanomicrobia, ",
        "Domain: Bacteria, Phylum: Actinobacteriota, Class: Actinobacteria, "
    ]
    property_seq = property_tokenizer(
        descriptions,
        return_tensors='pt',            # 返回 PyTorch tensor
        padding=True,                   # 自动 padding 到最长序列
        truncation=True,                # 超长截断
        max_length=512                  # 可选
    )

    property_seq = {k: v.to(property_encoder.device) for k, v in property_seq.items()}

    microbe = Microbe(aa_encoder, 33, property_encoder, trainable, cross_hidden_size=512)
    microbe = microbe.to(property_encoder.device)
    pred = microbe(aa_seq, property_seq)

    assert pred.shape == (4, 1)


def test_esm2_as_aaencoder():
    aa_encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    property_encoder = nn.Embedding(10, 512, padding_idx=0)
    trainable = {'aa_encoder': True, 'property_encoder': True, 'llm_decoder': True}

    aa_seq = torch.randint(0, 9, (4, 10))   # batch = 4, seq_len = 10
    property_seq = torch.randint(0, 9, (4, 10))
    model = Microbe(aa_encoder, 33, property_encoder, trainable, cross_hidden_size=512)
    pred = model(aa_seq, property_seq)

    assert pred.shape == (4, 1)