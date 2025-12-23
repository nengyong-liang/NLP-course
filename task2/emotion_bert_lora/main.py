from config import Config
from train import train_model
from evaluate import evaluate_model
from data_loader import get_dataloaders
from model import get_model_with_lora
from utils import plot_loss_curve, plot_confusion_matrix, plot_roc_curve
import torch
import os

def main():
    config = Config()
    
    # 1. 加载数据
    _, _, test_loader, _ = get_dataloaders(config)
    
    # 2. 训练模型（带 LoRA）
    print("🚀 Starting LoRA fine-tuning...")
    ft_model = train_model(config)
    
    # 3. 加载原始模型（无微调）
    _, base_model = get_model_with_lora(config)
    base_model.eval()
    
    # 4. 评估
    print("📊 Evaluating base model...")
    base_result = evaluate_model(base_model, test_loader, config, "Base Model (No Fine-tuning)")
    
    print("📊 Evaluating fine-tuned model...")
    ft_result = evaluate_model(ft_model, test_loader, config, "LoRA Fine-tuned Model")
    
    # 5. 绘图
    plot_loss_curve(config.train_log_path, config.plot_dir)
    
    # 混淆矩阵
    plot_confusion_matrix(
        ft_result["labels"], ft_result["preds"],
        os.path.join(config.plot_dir, "confusion_matrix.png")
    )
    
    # ROC 曲线（仅 fine-tuned）
    plot_roc_curve(
        ft_result["labels"], np.array(ft_result["probs"]),
        os.path.join(config.plot_dir, "roc_curve.png")
    )
    
    print("✅ All done! Results saved in ./logs/")

if __name__ == "__main__":
    main()