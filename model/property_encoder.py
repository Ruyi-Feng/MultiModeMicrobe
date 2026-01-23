from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.preprocessing import MinMaxScaler
import json
import numpy as np
import torch
import torch.nn as nn


# ====================用Qwen做description的property encoder====================
def get_llm_property_encoder(model_path="Qwen/Qwen-1_8B",
                         device="cuda",
                         pad_token="<|endoftext|>",
                         use_lora=False,
                         lora_r=64,
                         lora_alpha=16,
                         lora_dropout=0.05,
                         lora_target_modules=None):

    # 1. Tokenizer
    property_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Qwen 默认通常没有 pad_token，建议手动指定 eos 作为 pad
    if property_tokenizer.pad_token is None:
        property_tokenizer.pad_token = pad_token
        property_tokenizer.pad_token_id = property_tokenizer.convert_tokens_to_ids(pad_token)

    # 2. 回退到 AutoModelForCausalLM (为了解决报错)
    # 我们会在 forward 时通过 output_hidden_states=True 来获取表征
    property_encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if use_lora else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True if use_lora else False
    )

    if not use_lora:
        property_encoder = property_encoder.to(device)

    # 3. LoRA 配置
    if use_lora:
        if lora_target_modules is None:
            lora_target_modules = ["c_attn", "c_proj", "w1", "w2"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="lora_only",
            # 【注意】虽然我们做特征提取，但因为基础模型是 CausalLM，
            # 设为 CAUSAL_LM 可以避免 PEFT 报 mismatch 错误。
            # 我们只需要在训练逻辑中忽略 lm_head 的输出即可。
            task_type=TaskType.CAUSAL_LM,
        )

        property_encoder = get_peft_model(property_encoder, lora_config)

        property_encoder.gradient_checkpointing_enable()
        property_encoder.config.use_cache = False

        if hasattr(property_encoder, 'enable_input_require_grads'):
            property_encoder.enable_input_require_grads()

        print("=" * 50)
        print("Qwen CLIP Encoder Ready (Backbone only)")
        property_encoder.print_trainable_parameters()
        print("=" * 50)

    return property_encoder, property_tokenizer

def llm_property_converter(property_seq, property_tokenizer, device="cuda"):
    seq_with_cls = ["<|im_start|> " + seq + "<|endoftext|>" for seq in property_seq]
    property_seq = property_tokenizer(
        seq_with_cls,
        return_tensors='pt',            # 返回 PyTorch tensor
        padding=True,                   # 自动 padding 到最长序列
        truncation=True,                # 超长截断
        max_length=512                  # 可选
    )

    property_seq = {k: v.to(device) for k, v in property_seq.items()}
    tail_index = (property_seq['attention_mask'].sum(1) - 1)

    return property_seq, tail_index

# ================直接提取property的数值进入encoder===============
def get_numerical_property_encoder(property_dim,
                                   property_embedding_dim,
                                   device="cuda"):
    encoder = StrainEmbeddingGenerator(input_dim=property_dim, embedding_dim=property_embedding_dim)
    encoder.to(device)
    tokenizer = numerical_property_tokenizer

    return encoder, tokenizer

def numerical_property_tokenizer(property_seqs: list, device="cuda"):
    batch_vec = []
    for property_seq in property_seqs:
        if property_seq.endswith("\n"):
            property_seq = property_seq[:-1]
        property_seq = eval(property_seq)
        emb_vec = standardize_strain_features(property_seq)
        batch_vec.append(emb_vec)
    batch_vec = torch.tensor(batch_vec).to(device)
    return batch_vec  # [batch_size, 4]

def oxygen_tolerance_to_numeric(oxygen_type: str) -> float:
    """将氧气耐受度分类转为数值"""
    if oxygen_type is None:
        return 1.0
    oxygen_map = {
        "anaerobic": 0.0,          # 厌氧
        "facultative anaerobic": 1.0,  # 兼性厌氧
        "aerobic": 2.0             # 好氧
    }
    return oxygen_map.get(oxygen_type.lower(), 1.0)  # 默认兼性厌氧

def standardize_strain_features(strain_feature: dict) -> np.ndarray:
    # 这里需要增加如何处理none值
    """标准化菌株特征为4维向量（0-1范围）"""
    # 1. 提取并转换原始特征
    ph = strain_feature["pHOpt."]
    salt = strain_feature["naclopt."]
    oxygen = oxygen_tolerance_to_numeric(strain_feature["Oxygen Tolerance"])
    temp = strain_feature["Topt."]
    
    # 2. 定义各特征的合理取值范围（适配绝大多数菌株）
    scalers = {
        "ph": MinMaxScaler(feature_range=(0, 1)).fit([[1], [14]]),  # pH 1-14
        "salt": MinMaxScaler(feature_range=(0, 1)).fit([[0], [30]]), # 盐分 0-30%
        "oxygen": MinMaxScaler(feature_range=(0, 1)).fit([[0], [2]]),# 氧气 0-2
        "temp": MinMaxScaler(feature_range=(0, 1)).fit([[0], [100]]) # 温度 0-100℃
    }
    
    # 3. 标准化每个特征
    ph_norm = scalers["ph"].transform([[ph]])[0][0]
    salt_norm = scalers["salt"].transform([[salt]])[0][0]
    oxygen_norm = scalers["oxygen"].transform([[oxygen]])[0][0]
    temp_norm = scalers["temp"].transform([[temp]])[0][0]

    if np.isnan(ph_norm):
        ph_norm = -1
    if np.isnan(salt_norm):
        salt_norm = -1
    if np.isnan(oxygen_norm):
        oxygen_norm = -1
    if np.isnan(temp_norm):
        temp_norm = -1

    # 4. 返回4维标准化特征
    return np.array([ph_norm, salt_norm, oxygen_norm, temp_norm], dtype=np.float32)


class StrainEmbeddingGenerator(nn.Module):
    """可选：将4维特征扩展为固定高维embedding（如64维，适配深度学习）"""
    def __init__(self, input_dim=4, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.LayerNorm(embedding_dim)  # 归一化，提升稳定性
        )
    
    def forward(self, x):
        """输入：4维标准化特征（tensor），输出：embedding_dim维embedding"""
        return self.embedding_layer(x)
