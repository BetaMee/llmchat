import json
import torch
from typing import List, Dict, Any, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer
from loguru import logger


class LogisticsDataset:
    """物流填单数据集类"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_jsonl_data(self, filepath: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """加载JSONL格式的数据"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line.strip()))
        logger.info(f"从 {filepath} 加载了 {len(data)} 条数据")
        return data
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """格式化提示词"""
        prompt = f"<|im_start|>system\n你是一个专业的物流信息提取助手，能够从物流填单信息中准确提取结构化数据。<|im_end|>\n"
        prompt += f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{output}<|im_end|>"
        return prompt
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """分词函数"""
        # 格式化提示词
        prompts = [
            self.format_prompt(
                examples['instruction'][i],
                examples['input'][i],
                examples['output'][i]
            )
            for i in range(len(examples['instruction']))
        ]
        
        # 分词
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 设置标签（用于训练）
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def create_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        """创建HuggingFace数据集"""
        # 转换为Dataset格式
        dataset_dict = {
            'instruction': [item['instruction'] for item in data],
            'input': [item['input'] for item in data],
            'output': [item['output'] for item in data]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # 应用分词
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def load_and_process(self, filepath: str, max_samples: Optional[int] = None) -> Dataset:
        """加载并处理数据"""
        # 加载原始数据
        raw_data = self.load_jsonl_data(filepath, max_samples)
        
        # 创建数据集
        dataset = self.create_dataset(raw_data)
        
        return dataset


def create_data_collator(tokenizer: PreTrainedTokenizer):
    """创建数据整理器"""
    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 获取最大长度
        max_length = max(len(feature['input_ids']) for feature in features)
        
        # 填充到相同长度
        batch = {}
        for key in features[0].keys():
            batch[key] = []
            for feature in features:
                if key == 'input_ids':
                    padded = feature[key] + [tokenizer.pad_token_id] * (max_length - len(feature[key]))
                elif key == 'attention_mask':
                    padded = feature[key] + [0] * (max_length - len(feature[key]))
                elif key == 'labels':
                    padded = feature[key] + [-100] * (max_length - len(feature[key]))
                else:
                    padded = feature[key]
                batch[key].append(padded)
        
        # 转换为tensor
        for key in batch:
            batch[key] = torch.tensor(batch[key])
        
        return batch
    
    return data_collator


def load_datasets(config: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> tuple:
    """加载训练和验证数据集"""
    dataset_handler = LogisticsDataset(tokenizer, config['model']['max_length'])
    
    # 加载训练数据
    train_dataset = dataset_handler.load_and_process(
        config['data']['train_file'],
        config['data']['max_samples']
    )
    
    # 加载验证数据
    eval_dataset = dataset_handler.load_and_process(
        config['data']['eval_file'],
        config['data']['max_samples']
    )
    
    # 创建数据整理器
    data_collator = create_data_collator(tokenizer)
    
    logger.info(f"训练数据集大小: {len(train_dataset)}")
    logger.info(f"验证数据集大小: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset, data_collator 