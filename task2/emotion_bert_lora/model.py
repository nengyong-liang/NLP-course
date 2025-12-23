from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from config import Config

def get_model_with_lora(config: Config):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    )
    base_model.to(config.device)
    
    # ✅ BERT 的 attention 层命名是 query, key, value
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["query", "key", "value"],  # ← 关键修改！
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, base_model