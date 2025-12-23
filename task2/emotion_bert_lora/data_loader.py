import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from config import Config
import torch

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 原始数据已空格分词，需合并为字符串
        sentence = "".join(text.split())  # 去掉空格（中文无需空格）
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0, encoding='utf-8')
    # 处理可能的 NaN
    df = df.dropna().reset_index(drop=True)
    texts = df['text_a'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

def get_dataloaders(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    train_texts, train_labels = load_data(f"{config.data_dir}/train.tsv")
    dev_texts, dev_labels = load_data(f"{config.data_dir}/dev.tsv")
    test_texts, test_labels = load_data(f"{config.data_dir}/test.tsv")
    
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, config.max_length)
    dev_dataset = EmotionDataset(dev_texts, dev_labels, tokenizer, config.max_length)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, dev_loader, test_loader, tokenizer