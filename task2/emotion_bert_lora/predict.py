import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from config import Config
from data_loader import get_dataloaders, load_data
import textwrap

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_models_and_tokenizer(config: Config):
    """加载原始模型、微调模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 原始模型（无 LoRA）
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    ).to(config.device)
    base_model.eval()
    
    # 微调模型（LoRA）
    ft_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    )
    ft_model = PeftModel.from_pretrained(ft_model, os.path.join(config.best_model_path, ""))
    ft_model = ft_model.to(config.device)
    ft_model.eval()
    
    return base_model, ft_model, tokenizer

def predict(model, dataloader, config: Config):
    """通用预测函数"""
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrices(y_true, base_preds, ft_preds, save_dir):
    """绘制原始模型 vs 微调模型的混淆矩阵对比"""
    labels = ["消极", "中性", "积极"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (preds, title) in enumerate(zip([base_preds, ft_preds], ["原始模型", "LoRA 微调模型"])):
        cm = confusion_matrix(y_true, preds, labels=[0,1,2])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=axes[i])
        axes[i].set_title(f'{title}\n混淆矩阵', fontsize=14)
        axes[i].set_xlabel('预测标签')
        axes[i].set_ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_comparison.png'), dpi=300)
    plt.close()

def plot_roc_curve(y_true, base_probs, ft_probs, save_dir):
    """绘制多分类 ROC-AUC 曲线（一对多）"""
    y_bin = label_binarize(y_true, classes=[0,1,2])
    n_classes = 3
    colors = ['red', 'green', 'blue']
    labels = ['消极', '中性', '积极']
    
    plt.figure(figsize=(8, 6))
    
    # 微调模型 ROC
    for i, color, label in zip(range(n_classes), colors, labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], ft_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{label} (LoRA, AUC={roc_auc_val:.3f})')
    
    # 原始模型 ROC（虚线）
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], base_probs[:, i])
        plt.plot(fpr, tpr, color=color, lw=1, linestyle='--', alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('多分类 ROC 曲线（实线：微调；虚线：原始）')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()

def plot_probability_distribution(base_probs, ft_probs, save_dir):
    """绘制预测概率分布直方图（看模型置信度）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (probs, title) in enumerate(zip([base_probs, ft_probs], ["原始模型", "LoRA 微调模型"])):
        max_probs = np.max(probs, axis=1)
        axes[i].hist(max_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{title}：最大预测概率分布')
        axes[i].set_xlabel('最大预测概率')
        axes[i].set_ylabel('样本数量')
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prob_dist.png'), dpi=300)
    plt.close()

def save_error_analysis(test_texts, y_true, base_preds, ft_preds, save_path):
    """保存错误样本分析（展示典型错误）"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== 错误样本分析（仅展示微调模型仍出错的样本）===\n\n")
        count = 0
        for i, (text, true, base_pred, ft_pred) in enumerate(zip(test_texts, y_true, base_preds, ft_preds)):
            if true != ft_pred and count < 20:  # 只展示 20 个
                label_map = {0: "消极", 1: "中性", 2: "积极"}
                sentence = "".join(text.split())  # 还原为连续中文
                wrapped = "\n".join(textwrap.wrap(sentence, width=30))
                f.write(f"样本 #{count+1}:\n")
                f.write(f"  原句: {wrapped}\n")
                f.write(f"  真实: {label_map[true]}\n")
                f.write(f"  原始模型预测: {label_map[base_pred]}\n")
                f.write(f"  微调模型预测: {label_map[ft_pred]}\n")
                f.write("-" * 50 + "\n")
                count += 1

def plot_performance_radar(base_metrics, ft_metrics, save_dir):
    """绘制性能指标雷达图"""
    labels = np.array(['Accuracy', 'Precision', 'Recall', 'F1'])
    base_vals = [base_metrics['acc'], base_metrics['prec'], base_metrics['rec'], base_metrics['f1']]
    ft_vals = [ft_metrics['acc'], ft_metrics['prec'], ft_metrics['rec'], ft_metrics['f1']]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    base_vals += base_vals[:1]
    ft_vals += ft_vals[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, base_vals, color='red', alpha=0.25, label='原始模型')
    ax.fill(angles, ft_vals, color='blue', alpha=0.25, label='LoRA 微调')
    ax.plot(angles, base_vals, color='red', linewidth=2)
    ax.plot(angles, ft_vals, color='blue', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title('模型性能雷达图', size=16, y=1.08)
    plt.savefig(os.path.join(save_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    config = Config()
    config.data_dir = os.path.join(config.data_dir, "")  # 可改为全量数据路径
    
    # 加载数据和模型
    _, _, test_loader, tokenizer = get_dataloaders(config)
    test_texts, _ = load_data(os.path.join(config.data_dir, "test.tsv"))
    
    base_model, ft_model, _ = load_models_and_tokenizer(config)
    
    # 预测
    y_true, base_preds, base_probs = predict(base_model, test_loader, config)
    _, ft_preds, ft_probs = predict(ft_model, test_loader, config)
    
    # 计算指标
    def compute_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
    
    base_metrics = compute_metrics(y_true, base_preds)
    ft_metrics = compute_metrics(y_true, ft_preds)
    
    # 打印指标
    print("📊 性能对比:")
    print(f"原始模型 → Acc: {base_metrics['acc']:.4f}, F1: {base_metrics['f1']:.4f}")
    print(f"LoRA微调 → Acc: {ft_metrics['acc']:.4f}, F1: {ft_metrics['f1']:.4f}")
    
    # 可视化
    plot_confusion_matrices(y_true, base_preds, ft_preds, config.plot_dir)
    plot_roc_curve(y_true, base_probs, ft_probs, config.plot_dir)
    plot_probability_distribution(base_probs, ft_probs, config.plot_dir)
    plot_performance_radar(base_metrics, ft_metrics, config.plot_dir)
    save_error_analysis(test_texts, y_true, base_preds, ft_preds, 
                        os.path.join(config.plot_dir, "error_samples.txt"))
    
    print(f"✅ 所有可视化结果已保存至: {config.plot_dir}")

if __name__ == "__main__":
    main()