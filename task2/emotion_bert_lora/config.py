import torch

class Config:
    # 数据路径
    data_dir = r"D:\1_LNY\code\0_data\NLP-course\task2\mini"
    
    # 模型
    model_name = "bert-base-chinese"  # 中文 DistilBERT
    num_labels = 3
    max_length = 128
    
    # 训练参数
    batch_size = 32
    epochs = 2
    learning_rate = 2e-5
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 日志
    log_dir = "./logs"
    train_log_path = "./logs/train_log.csv"
    eval_result_path = "./logs/eval_results.txt"
    plot_dir = "./logs/plots"
    best_model_path = "./logs/final_lora_model"
    
    # 确保日志目录存在
    import os
    os.makedirs(plot_dir, exist_ok=True)