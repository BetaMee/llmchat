import torch
from typing import Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from loguru import logger


def setup_quantization_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """设置量化配置"""
    if config['model']['use_4bit']:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=config['model']['use_nested_quant'],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config['model']['bnb_4bit_compute_dtype'])
        )
    return None


def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """设置LoRA配置"""
    return LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type'],
    )


def setup_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """设置训练参数"""
    return TrainingArguments(
        output_dir=config['output']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        logging_steps=config['training']['logging_steps'],
        save_total_limit=config['output']['save_total_limit'],
        load_best_model_at_end=config['output']['load_best_model_at_end'],
        metric_for_best_model=config['output']['metric_for_best_model'],
        greater_is_better=False,  # 对于loss，越小越好
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        fp16=config['device']['use_fp16'],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["wandb"] if config['logging']['use_wandb'] else [],
    )


def load_model_and_tokenizer(config: Dict[str, Any]):
    """加载模型和分词器"""
    logger.info(f"正在加载基础模型: {config['model']['base_model']}")
    
    # 设置量化配置
    quantization_config = setup_quantization_config(config)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model'],
        trust_remote_code=True
    )
    
    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if config['device']['use_fp16'] else torch.float32
    )
    
    # 准备模型进行kbit训练
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    # 设置LoRA
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    logger.info("模型和分词器加载完成")
    
    return model, tokenizer


def setup_device(config: Dict[str, Any]):
    """设置设备"""
    if config['device']['use_cpu']:
        device = "cpu"
    elif config['device']['use_mps'] and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"使用设备: {device}")
    return device 