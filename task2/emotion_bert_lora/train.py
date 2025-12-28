import torch
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from config import Config
from model import get_model_with_lora
from data_loader import get_dataloaders
from evaluate import evaluate_model
import pdb

def train_model(config: Config):
    train_loader, dev_loader, _, _ = get_dataloaders(config)
    model, _ = get_model_with_lora(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    log_data = []
    
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估训练集
        avg_loss = total_loss / len(train_loader) #该loss是训练集的loss
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # 评估验证集
        eval_metrics = evaluate_model(model, dev_loader, config, model_name="lora")
        print(f"Evaluation Metrics: {eval_metrics['accuracy']}")

        # 保存训练日志 记录训练损失和评估指标
        log_data.append({"epoch": epoch + 1, "loss": avg_loss,"eval_metrics_accuracy": eval_metrics['accuracy']})
        
        # pdb.set_trace()
        pd.DataFrame(log_data).to_csv(config.train_log_path, index=False)
        # 每5个epoch保存一次模型，和当前模型的指标
        # if (epoch + 1) % 5 == 0:
        model.save_pretrained(f"./logs/epoch_{epoch+1}_lora_model")

    # 保存训练日志
    pd.DataFrame(log_data).to_csv(config.train_log_path, index=False)
    # 可选：保存模型
    model.save_pretrained("./logs/final_lora_model")
    return model