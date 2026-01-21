# ======================== 三个无描述、名称为英文的培养基样本 ========================
# 样本1：大肠杆菌LB培养基（富有机，含无SMILES的复杂有机物）
medium_sample_lb = {
    "components": [
        # 无SMILES的复杂有机成分（核心）
        {"smiles": "", "name": "Yeast Extract", "is_organic": True, "concentration_gL": 5.0},
        {"smiles": "", "name": "Tryptone", "is_organic": True, "concentration_gL": 10.0},
        # 无机成分
        {"smiles": "NaCl", "name": "Sodium Chloride", "is_organic": False, "concentration_gL": 10.0},
        # 缓冲无机成分
        {"smiles": "K2HPO4", "name": "Dipotassium Hydrogen Phosphate", "is_organic": False, "concentration_gL": 1.0}
    ]
}

# 样本2：硝化细菌自养培养基（无机为主，少量有机碳源）
medium_sample_autotrophic = {
    "components": [
        # 无机氮源（核心，自养菌能源）
        {"smiles": "NaNO2", "name": "Sodium Nitrite", "is_organic": False, "concentration_gL": 2.0},
        {"smiles": "NH4Cl", "name": "Ammonium Chloride", "is_organic": False, "concentration_gL": 1.0},
        # 无机磷/钾源
        {"smiles": "KH2PO4", "name": "Potassium Dihydrogen Phosphate", "is_organic": False, "concentration_gL": 1.0},
        {"smiles": "MgSO4", "name": "Magnesium Sulfate", "is_organic": False, "concentration_gL": 0.2},
        # 少量有机碳源（辅助）
        {"smiles": "C6H12O6", "name": "Glucose", "is_organic": True, "concentration_gL": 0.5}
    ]
}

# 样本3：大肠杆菌M9合成培养基（纯化学组分，全有SMILES）
medium_sample_m9 = {
    "components": [
        # 有机碳源（核心）
        {"smiles": "C6H12O6", "name": "Glucose", "is_organic": True, "concentration_gL": 4.0},
        # 无机氮源
        {"smiles": "NH4Cl", "name": "Ammonium Chloride", "is_organic": False, "concentration_gL": 1.0},
        # 无机磷源
        {"smiles": "Na2HPO4", "name": "Disodium Hydrogen Phosphate", "is_organic": False, "concentration_gL": 6.0},
        {"smiles": "KH2PO4", "name": "Potassium Dihydrogen Phosphate", "is_organic": False, "concentration_gL": 3.0},
        # 无机盐
        {"smiles": "NaCl", "name": "Sodium Chloride", "is_organic": False, "concentration_gL": 0.5},
        {"smiles": "MgSO4", "name": "Magnesium Sulfate", "is_organic": False, "concentration_gL": 0.24}
    ]
}

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict

# ======================== 基础配置 ========================
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

# 1. ChemBERTa模型（处理SMILES，适配有机/无机化合物）
chem_model_name = "seyonec/ChemBERTa-zinc-base-v1"
chem_tokenizer = AutoTokenizer.from_pretrained(chem_model_name)
chem_model = AutoModel.from_pretrained(chem_model_name).to(device)

# 2. 中文BERT模型（处理无SMILES的英文名称）
text_model_name = "bert-base-chinese"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name).to(device)

# ======================== 工具函数（适配无description、英文名称） ========================
def get_single_component_embedding(component: Dict) -> np.ndarray:
    """单个化合物embedding：1维浓度 + 768维特征（769维）"""
    # 1. 浓度标准化（0-1范围，适配0-100 g/L）
    conc = component["concentration_gL"]
    conc_scaler = MinMaxScaler(feature_range=(0, 1))
    conc_scaler.fit([[0], [100]])  # 培养基成分常见浓度范围
    norm_conc = conc_scaler.transform([[conc]])[0][0]  # 1维标准化浓度
    
    # 2. 生成化合物的分子/文本特征（768维）
    smiles = component.get("smiles", "").strip()
    name = component.get("name", "")
    try:
        if smiles:
            # 有SMILES：ChemBERTa处理（有机/无机化合物均适用）
            inputs = chem_tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = chem_model(**inputs)
            comp_emb = outputs.last_hidden_state[0][0].cpu().numpy()
        else:
            # 无SMILES：仅用英文名称生成文本特征
            text = f"Name: {name}"
            inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = text_model(**inputs)
            comp_emb = outputs.last_hidden_state[0][0].cpu().numpy()
    except Exception as e:
        print(f"Processing {name} failed: {e}")
        comp_emb = np.zeros(768, dtype=np.float32)  # 失败时返回全0
    
    # 3. 拼接：浓度（1维） + 化合物特征（768维）
    single_emb = np.concatenate([[norm_conc], comp_emb], axis=0)
    return single_emb.astype(np.float32)

def get_medium_comprehensive_embedding(medium_data: Dict) -> np.ndarray:
    """
    生成培养基综合embedding：
    1. 对每个化合物生成（浓度+特征）embedding（769维/个）
    2. 按浓度加权平均所有成分的embedding，得到最终综合embedding
    输出维度：769维
    """
    components = medium_data["components"]
    component_embs = []
    weights = []
    
    # 1. 生成每个成分的（浓度+特征）embedding
    for comp in components:
        if comp["concentration_gL"] <= 0:  # 跳过浓度为0的无效成分
            continue
        single_emb = get_single_component_embedding(comp)
        component_embs.append(single_emb)
        weights.append(comp["concentration_gL"])  # 按浓度加权
    
    # 2. 无有效成分时返回全0
    if len(component_embs) == 0:
        return np.zeros(769, dtype=np.float32)
    
    # 3. 浓度加权平均（高浓度成分占更高权重）
    weights = np.array(weights) / sum(weights)
    comprehensive_emb = np.sum(component_embs * weights[:, np.newaxis], axis=0)
    
    # 4. 归一化（避免数值范围差异）
    if np.linalg.norm(comprehensive_emb) > 0:
        comprehensive_emb = comprehensive_emb / np.linalg.norm(comprehensive_emb)
    
    return comprehensive_emb

# ======================== 测试三个样本 ========================
if __name__ == "__main__":
    # 生成三个样本的综合embedding
    emb_lb = get_medium_comprehensive_embedding(medium_sample_lb)
    emb_autotrophic = get_medium_comprehensive_embedding(medium_sample_autotrophic)
    emb_m9 = get_medium_comprehensive_embedding(medium_sample_m9)
    
    # 输出每个样本的核心信息
    def print_sample_info(name, sample, emb):
        print(f"\n=== {name} ===")
        # 统计有机/无机占比
        total_conc = sum([c["concentration_gL"] for c in sample["components"]])
        organic_conc = sum([c["concentration_gL"] for c in sample["components"] if c["is_organic"]])
        organic_ratio = (organic_conc / total_conc) * 100 if total_conc != 0 else 0
        # 输出关键信息
        print(f"Organic component ratio: {np.round(organic_ratio, 2)}%")
        print(f"Comprehensive embedding dimension: {len(emb)}")
        print(f"First 3 values of embedding: {np.round(emb[:3], 4)}")
        print(f"Last 3 values of embedding: {np.round(emb[-3:], 4)}")
    
    print_sample_info("E. coli LB Medium", medium_sample_lb, emb_lb)
    print_sample_info("Nitrifying Bacteria Autotrophic Medium", medium_sample_autotrophic, emb_autotrophic)
    print_sample_info("E. coli M9 Synthetic Medium", medium_sample_m9, emb_m9)
    
    # 计算样本间余弦相似度（可选，用于对比样本差异）
    from sklearn.metrics.pairwise import cosine_similarity
    sim_lb_autotrophic = cosine_similarity([emb_lb], [emb_autotrophic])[0][0]
    sim_lb_m9 = cosine_similarity([emb_lb], [emb_m9])[0][0]
    sim_autotrophic_m9 = cosine_similarity([emb_autotrophic], [emb_m9])[0][0]
    
    print("\n=== Cosine Similarity Between Samples ===")
    print(f"LB vs Autotrophic: {np.round(sim_lb_autotrophic, 4)}")
    print(f"LB vs M9: {np.round(sim_lb_m9, 4)}")
    print(f"Autotrophic vs M9: {np.round(sim_autotrophic_m9, 4)}")