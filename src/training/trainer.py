import os
import yaml
import torch
from typing import Dict, Any
from transformers import Trainer
from loguru import logger
from src.model.model_setup import load_model_and_tokenizer, setup_training_arguments, setup_device
from src.data_processing.dataset import load_datasets


class LogisticsTrainer:
    """物流信息提取模型训练器"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.device = setup_device(self.config)
        
        # 设置日志
        logger.add(
            f"{self.config['output']['output_dir']}/training.log",
            rotation="10 MB",
            level=self.config['logging']['log_level']
        )
        
        # 设置wandb
        if self.config['logging']['use_wandb']:
            import wandb
            wandb.init(
                project=self.config['logging']['wandb_project'],
                name=self.config['model']['model_name']
            )
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载完成: {config_path}")
        return config
    
    def prepare_training(self):
        """准备训练环境"""
        logger.info("开始准备训练环境...")
        
        # 创建输出目录
        os.makedirs(self.config['output']['output_dir'], exist_ok=True)
        
        # 加载模型和分词器
        self.model, self.tokenizer = load_model_and_tokenizer(self.config)
        
        # 加载数据集
        self.train_dataset, self.eval_dataset, self.data_collator = load_datasets(
            self.config, self.tokenizer
        )
        
        # 设置训练参数
        self.training_args = setup_training_arguments(self.config)
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("训练环境准备完成")
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        try:
            # 开始训练
            train_result = self.trainer.train()
            
            # 保存最终模型
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config['output']['output_dir'])
            
            # 保存训练结果
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
            
            logger.info(f"训练完成！训练指标: {metrics}")
            
            # 评估模型
            self.evaluate()
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
    
    def evaluate(self):
        """评估模型"""
        logger.info("开始评估模型...")
        
        try:
            # 在验证集上评估
            metrics = self.trainer.evaluate()
            
            # 记录评估结果
            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)
            
            logger.info(f"评估完成！评估指标: {metrics}")
            
        except Exception as e:
            logger.error(f"评估过程中出现错误: {e}")
            raise
    
    def save_model_info(self):
        """保存模型信息"""
        model_info = {
            "model_name": self.config['model']['model_name'],
            "base_model": self.config['model']['base_model'],
            "training_config": self.config,
            "model_type": "logistics_extractor",
            "task": "information_extraction",
            "language": "zh"
        }
        
        info_path = os.path.join(self.config['output']['output_dir'], "model_info.json")
        import json
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型信息已保存到: {info_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="物流信息提取模型训练")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = LogisticsTrainer(args.config)
    
    # 准备训练
    trainer.prepare_training()
    
    # 开始训练
    trainer.train()
    
    # 保存模型信息
    trainer.save_model_info()
    
    logger.info("训练流程完成！")


if __name__ == "__main__":
    main() 