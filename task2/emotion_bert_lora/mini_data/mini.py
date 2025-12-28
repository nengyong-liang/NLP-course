import os
import pandas as pd
from pathlib import Path

# === 配置 ===
DATA_DIR = r"D:\1_LNY\code\0_data\NLP-course\task2"
MINI_DIR = os.path.join(DATA_DIR, "mini")
N_SAMPLES_PER_LABEL = 20  # 每个情绪类别抽取 20 条

# 确保 mini 目录存在
Path(MINI_DIR).mkdir(parents=True, exist_ok=True)

def create_mini_split(filename: str):
    """对单个 tsv 文件进行分层采样并保存到 mini 目录"""
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"⚠️ {file_path} 不存在，跳过。")
        return
    
    # 读取数据
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    df = df.dropna().reset_index(drop=True)
    
    # 分层采样：对每个 label 抽取 N_SAMPLES_PER_LABEL 条（若不足则全取）
    sampled_dfs = []
    for label in [0, 1, 2]:
        label_df = df[df['label'] == label]
        if len(label_df) == 0:
            continue
        # 随机抽取（可重复运行时结果不同，如需固定加 random_state=42）
        n_sample = min(N_SAMPLES_PER_LABEL, len(label_df))
        sampled = label_df.sample(n=n_sample, random_state=42)  # 固定随机种子便于复现
        sampled_dfs.append(sampled)
    
    # 合并并保存
    mini_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱顺序
    mini_df.to_csv(os.path.join(MINI_DIR, filename), sep='\t', index=False, encoding='utf-8')
    print(f"✅ {filename}: 原始 {len(df)} 条 → 采样 {len(mini_df)} 条")

# === 主程序 ===
if __name__ == "__main__":
    for split in ["train.tsv", "dev.tsv", "test.tsv"]:
        create_mini_split(split)
    
    print(f"\n🎉 小型调试数据集已保存至：{MINI_DIR}")