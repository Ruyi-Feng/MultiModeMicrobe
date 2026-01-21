import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA  # 可选：降维/升维工具
import torch
import torch.nn as nn

# ======================== 1. 定义典型菌株的特性样本（真实/合理数值） ========================
strain_samples = {
    # 大肠杆菌（E. coli）：兼性厌氧、适中性pH、中等盐分耐受、37℃最适
    "Escherichia coli": {
        "ph_opt": 7.0,          # 最适pH
        "salt_tolerance": 1.0,  # 耐受NaCl浓度（%）
        "oxygen_tolerance": "facultative anaerobic",  # 氧气耐受度：兼性厌氧
        "temp_opt": 37.0        # 最适温度（℃）
    },
    # 硝化细菌（Nitrifying bacteria）：好氧、弱碱性pH、低盐分耐受、28℃最适
    "Nitrifying bacteria": {
        "ph_opt": 7.8,
        "salt_tolerance": 0.5,
        "oxygen_tolerance": "aerobic",  # 好氧
        "temp_opt": 28.0
    },
    # 乳酸菌（Lactic acid bacteria）：厌氧、酸性pH、高盐分耐受、30℃最适
    "Lactic acid bacteria": {
        "ph_opt": 5.5,
        "salt_tolerance": 5.0,
        "oxygen_tolerance": "anaerobic",  # 厌氧
        "temp_opt": 30.0
    }
}

# ======================== 2. 核心工具函数：菌株特性→Embedding ========================
def oxygen_tolerance_to_numeric(oxygen_type: str) -> float:
    """将氧气耐受度分类转为数值"""
    oxygen_map = {
        "anaerobic": 0.0,          # 厌氧
        "facultative anaerobic": 1.0,  # 兼性厌氧
        "aerobic": 2.0             # 好氧
    }
    return oxygen_map.get(oxygen_type.lower(), 1.0)  # 默认兼性厌氧

def standardize_strain_features(strain_feature: dict) -> np.ndarray:
    """标准化菌株特征为4维向量（0-1范围）"""
    # 1. 提取并转换原始特征
    ph = strain_feature["ph_opt"]
    salt = strain_feature["salt_tolerance"]
    oxygen = oxygen_tolerance_to_numeric(strain_feature["oxygen_tolerance"])
    temp = strain_feature["temp_opt"]
    
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
    
    # 4. 返回4维标准化特征
    return np.array([ph_norm, salt_norm, oxygen_norm, temp_norm], dtype=np.float32)

class StrainEmbeddingGenerator(nn.Module):
    """可选：将4维特征扩展为固定高维embedding（如64维，适配深度学习）"""
    def __init__(self, input_dim=4, embedding_dim=64):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.LayerNorm(embedding_dim)  # 归一化，提升稳定性
        )
    
    def forward(self, x):
        """输入：4维标准化特征（tensor），输出：embedding_dim维embedding"""
        return self.embedding_layer(x)

# ======================== 3. 生成菌株Embedding ========================
if __name__ == "__main__":
    # 3.1 生成基础4维标准化embedding
    print("=== 基础4维标准化菌株Embedding ===")
    strain_4d_embs = {}
    for strain_name, features in strain_samples.items():
        emb_4d = standardize_strain_features(features)
        strain_4d_embs[strain_name] = emb_4d
        print(f"\n{strain_name}:")
        print(f"  原始特征：pH={features['ph_opt']}, 盐分={features['salt_tolerance']}%, 氧气={features['oxygen_tolerance']}, 温度={features['temp_opt']}℃")
        print(f"  4维Embedding：{np.round(emb_4d, 4)}")
    
    # 3.2 可选：生成64维高维embedding（适配深度学习）
    print("\n=== 扩展64维菌株Embedding（前5值） ===")
    generator = StrainEmbeddingGenerator(input_dim=4, embedding_dim=64)
    for strain_name, emb_4d in strain_4d_embs.items():
        # 转换为tensor输入模型
        x = torch.tensor(emb_4d, dtype=torch.float32).unsqueeze(0)
        emb_64d = generator(x).detach().numpy()[0]
        print(f"{strain_name} 64维Embedding前5值：{np.round(emb_64d[:5], 4)}")
    
    # 3.3 可选：计算菌株间相似度（余弦相似度，衡量特性差异）
    from sklearn.metrics.pairwise import cosine_similarity
    print("\n=== 菌株特性相似度（余弦相似度，越接近1越相似） ===")
    strain_names = list(strain_4d_embs.keys())
    for i in range(len(strain_names)):
        for j in range(i+1, len(strain_names)):
            s1 = strain_names[i]
            s2 = strain_names[j]
            sim = cosine_similarity([strain_4d_embs[s1]], [strain_4d_embs[s2]])[0][0]
            print(f"{s1} vs {s2}: {np.round(sim, 4)}")