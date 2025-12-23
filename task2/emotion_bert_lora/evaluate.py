import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix
import pandas as pd
from config import Config
from data_loader import get_dataloaders
from model import get_model_with_lora
import os

def evaluate_model(model, dataloader, config: Config, model_name="fine-tuned"):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 多分类 AUC（需 one-hot）
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(all_labels, classes=[0,1,2])
    auc = roc_auc_score(y_bin, np.array(all_probs), average='macro', multi_class='ovr')
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # 保存结果
    with open(config.eval_result_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {model_name} ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (weighted): {f1:.4f}\n")
        f.write(f"AUC (macro): {auc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write("\n" + classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"]) + "\n")
    
    return {
        "model": model_name,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "labels": all_labels,
        "probs": all_probs,
        "preds": all_preds
    }