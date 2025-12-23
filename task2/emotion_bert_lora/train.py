import torch
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from config import Config
from model import get_model_with_lora
from data_loader import get_dataloaders

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
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        log_data.append({"epoch": epoch + 1, "loss": avg_loss})
    
    # 保存训练日志
    pd.DataFrame(log_data).to_csv(config.train_log_path, index=False)
    # 可选：保存模型
    model.save_pretrained("./logs/final_lora_model")
    return model