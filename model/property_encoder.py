import ast
import json

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


# 反归一化和 loss mask 都依赖这套范围。
# 这里收紧到接近真实分布的区间（旧版用极端边界 [1,14]/[0,100]/[0,30] 会把所有有效值挤到 [0.3,0.6]，
# 导致 MSE 信号过弱），新的范围让动态范围占满 [0,1]。
PROPERTY_RANGES = {
    "ph": (4.0, 10.0),
    "temp": (0.0, 80.0),
    "nacl": (0.0, 15.0),
    "oxygen": (0.0, 2.0),
}
# 缺失值哨兵；下游 loss 必须用 != MISSING_VALUE 做 mask
MISSING_VALUE = -1.0


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

    return encoder, tokenizer   # tokenizer的第1维和seq的维度保持一致

def _parse_property_string(s: str):
    """容错解析：先 ast.literal_eval（Python 字面量），失败则 json.loads（JSON 真假空），
    再失败则把 JSON 关键字替换后重试。返回 dict 或 None（None 表示解析失败）。"""
    s = s.strip()
    if not s:
        return None
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, dict) else None
    except (ValueError, SyntaxError):
        pass
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        s2 = (s.replace("null", "None")
                .replace("true", "True")
                .replace("false", "False"))
        v = ast.literal_eval(s2)
        return v if isinstance(v, dict) else None
    except (ValueError, SyntaxError):
        return None


# 解析失败 warning 只打一次，避免训练日志被淹
_PARSE_WARNED = {"flag": False}


def numerical_property_tokenizer(property_seqs: list, device="cuda", property_list=["ph", "temp", "nacl", "oxygen"]):
    import warnings

    batch_vec = []
    n_fail = 0
    first_fail_sample = None
    for property_seq in property_seqs:
        if isinstance(property_seq, str):
            parsed = _parse_property_string(property_seq.rstrip("\n"))
            if parsed is None:
                n_fail += 1
                if first_fail_sample is None:
                    first_fail_sample = property_seq[:200]
                parsed = {}
        elif isinstance(property_seq, dict):
            parsed = property_seq
        else:
            parsed = {}
        emb_vec = standardize_strain_features(parsed, property_list)
        batch_vec.append(emb_vec)

    if n_fail > 0 and not _PARSE_WARNED["flag"]:
        _PARSE_WARNED["flag"] = True
        warnings.warn(
            f"[numerical_property_tokenizer] {n_fail}/{len(property_seqs)} sample(s) "
            f"failed to parse on this batch. First failing sample (truncated):\n"
            f"  {first_fail_sample!r}\n"
            f"This will cause masked MSE to be 0 if it happens for every batch. "
            f"Check property.bin format vs extract_property_number key names."
        )

    batch_vec = torch.tensor(np.stack(batch_vec, axis=0), dtype=torch.float32).to(device)
    return batch_vec  # [batch_size, len(property_list)]

def oxygen_tolerance_to_numeric(oxygen_type):
    """字符串氧气分类转数值；不能识别时返回 None（让上游按缺失处理）。"""
    if oxygen_type is None:
        return None
    if isinstance(oxygen_type, (int, float)) and np.isfinite(oxygen_type):
        return float(oxygen_type)
    if not isinstance(oxygen_type, str):
        return None
    oxygen_map = {
        "anaerobic": 0.0,              # 厌氧
        "facultative anaerobic": 1.0,  # 兼性厌氧
        "aerobic": 2.0,                # 好氧
    }
    return oxygen_map.get(oxygen_type.lower().strip())

def extract_property_number(strain_feature: dict) -> dict:
    if "culture_pH_optimum" in strain_feature:
        ph = strain_feature["culture_pH_optimum"]
    else:
        ph = None
    if "culture_temp_optimum" in strain_feature:
        temp = strain_feature["culture_temp_optimum"]
    else:
        temp = None
    if "NaCl_optimum" in strain_feature:
        nacl = strain_feature["NaCl_optimum"]
    else:
        nacl = None
    if "Oxygen Tolerance" in strain_feature:
        oxygen = strain_feature["Oxygen Tolerance"]
    else:
        oxygen = None
    return ph, temp, nacl, oxygen

def _scale_or_missing(value, lo, hi):
    """把 value 线性 scale 到 [0,1]；缺失或非法返回 MISSING_VALUE。"""
    if value is None:
        return MISSING_VALUE
    try:
        v = float(value)
    except (TypeError, ValueError):
        return MISSING_VALUE
    if not np.isfinite(v):
        return MISSING_VALUE
    scaled = (v - lo) / (hi - lo)
    return float(np.clip(scaled, 0.0, 1.0))


def standardize_strain_features(strain_feature: dict, property_list=["ph", "temp", "nacl", "oxygen"]) -> np.ndarray:
    """标准化菌株特征为 0-1 向量；缺失值用 MISSING_VALUE (-1) 占位。

    下游 loss/metric 必须用 (target != MISSING_VALUE) 做 mask，否则
    缺失样本会以 -1 的标签污染 MSE，把模型拉成一条直线。
    """
    if not isinstance(strain_feature, dict):
        strain_feature = {}

    ph, temp, nacl, oxygen = extract_property_number(strain_feature)
    oxygen_num = oxygen_tolerance_to_numeric(oxygen)

    raw = {
        "ph": ph,
        "temp": temp,
        "nacl": nacl,
        "oxygen": oxygen_num,
    }

    feature_list = []
    for name in property_list:
        lo, hi = PROPERTY_RANGES[name]
        feature_list.append(_scale_or_missing(raw.get(name), lo, hi))
    return np.array(feature_list, dtype=np.float32)


def denormalize_property(values, name):
    """把 [0,1] 反归一化到原始物理量纲。values: torch.Tensor 或 np.ndarray。"""
    lo, hi = PROPERTY_RANGES[name]
    return values * (hi - lo) + lo


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
