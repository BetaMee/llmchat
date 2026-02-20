#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsloth + Qwen 微调训练脚本
用于训练 Figma JSON 生成模型
"""

import os
import torch
import yaml

# ⚠️ 重要：Unsloth 必须在 transformers, trl, peft 之前导入以启用所有优化
from unsloth import FastLanguageModel

# 然后导入其他训练相关库
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

class FigmaJSONTrainer:
    """Figma JSON 微调训练器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def check_unsloth_config(self):
        """检查 Unsloth 的配置状态"""
        print("\n" + "="*60)
        print("   Unsloth 2026.2.1 配置检查")
        print("="*60)
        
        # 检查环境变量
        print(f"\n环境变量:")
        print(f"  UNSLOTH_ENABLE_CCE: {os.environ.get('UNSLOTH_ENABLE_CCE', '未设置')}")
        print(f"  UNSLOTH_DISABLE_FUSED_LOSS: {os.environ.get('UNSLOTH_DISABLE_FUSED_LOSS', '未设置')}")
        print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '未设置')}")
        
        # 检查设备类型
        try:
            from unsloth_zoo.device_type import DEVICE_TYPE
            print(f"\n设备信息:")
            print(f"  设备类型: {DEVICE_TYPE}")
        except ImportError:
            print(f"\n设备信息: 无法获取 DEVICE_TYPE")
        
        # 检查显存
        if torch.cuda.is_available():
            print(f"\n显存状态:")
            print(f"  GPU 已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"  GPU 已预留: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            print(f"  GPU 总显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        
        # 检查模型配置
        if self.model is not None:
            print(f"\n模型信息:")
            print(f"  模型类型: {type(self.model).__name__}")
            if hasattr(self.model, 'config'):
                config_attrs = ['disable_fused_loss', 'use_cache']
                for attr in config_attrs:
                    if hasattr(self.model.config, attr):
                        print(f"  {attr}: {getattr(self.model.config, attr)}")
        
        print("="*60 + "\n")
    
    def load_model(self):
        """加载模型和分词器"""
        print("正在加载模型...")
        
        model_config = self.config['model']
        lora_config = self.config['lora']
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['name'],
            max_seq_length=model_config['max_seq_length'],
            dtype=model_config['dtype'],
            load_in_4bit=model_config['load_in_4bit'],
            load_in_8bit=model_config['load_in_8bit'],
            # device_map = "auto", 
            # max_memory = {0: "10GiB", "cpu": "20GiB"},  # 限制GPU使用10GB，其余放CPU
        )

        # 配置 LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config['r'],
            target_modules=lora_config['target_modules'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            use_gradient_checkpointing=lora_config['use_gradient_checkpointing'],
            random_state=lora_config['random_state'],
            use_rslora=lora_config['use_rslora'],
            loftq_config=lora_config['loftq_config'],
        )
        
        print("模型加载完成!")
        print(f"  模型: {model_config['name']}")
        print(f"  LoRA rank: {lora_config['r']}")
        print(f"  最大序列长度: {model_config['max_seq_length']}")
        print(f"  8-bit 量化: {model_config['load_in_8bit']}")

        # 检查 Unsloth 配置状态
        self.check_unsloth_config()
    
    def load_datasets(self):
        """加载数据集"""
        print("\n正在加载数据集...")
        
        data_config = self.config['data']
        
        # 加载训练集
        if os.path.exists(data_config['train_file']):
            self.train_dataset = load_dataset(
                'json',
                data_files=data_config['train_file'],
                split='train'
            )
            print(f"训练集样本数: {len(self.train_dataset)}")
        else:
            raise FileNotFoundError(f"训练文件不存在: {data_config['train_file']}")
        
        # 加载验证集
        if os.path.exists(data_config['val_file']):
            self.eval_dataset = load_dataset(
                'json',
                data_files=data_config['val_file'],
                split='train'
            )
            print(f"验证集样本数: {len(self.eval_dataset)}")
        
        # 显示示例
        print("\n训练样本示例:")
        sample_text = self.train_dataset[0]['text']
        preview_len = min(200, len(sample_text))
        print(sample_text[:preview_len] + "...")
    
    def prepare_training_args(self):
        """准备训练参数"""
        train_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=train_config['output_dir'],
            per_device_train_batch_size=train_config['per_device_train_batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            warmup_steps=train_config['warmup_steps'],
            num_train_epochs=train_config['num_train_epochs'],
            learning_rate=train_config['learning_rate'],
            fp16=train_config['fp16'],
            bf16=train_config['bf16'],
            logging_steps=train_config['logging_steps'],
            optim=train_config['optim'],
            weight_decay=train_config['weight_decay'],
            lr_scheduler_type=train_config['lr_scheduler_type'],
            seed=train_config['seed'],
            save_strategy=train_config['save_strategy'],
            save_steps=train_config['save_steps'],
            save_total_limit=train_config['save_total_limit'],
            report_to=train_config['report_to'],
            # 新版本使用 eval_strategy 而非 evaluation_strategy
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=train_config['save_steps'] if self.eval_dataset else None,
            load_best_model_at_end=True if self.eval_dataset else False,
        )
        
        return training_args
    
    def train(self):
        """开始训练"""
        print("\n准备训练...")
        
        training_args = self.prepare_training_args()
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config['model']['max_seq_length'],
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        
        print("\n开始训练...")
        trainer.train()
        
        print("\n训练完成!")
        
        # 保存模型
        self.save_models(trainer)
        
        return trainer
    
    def save_models(self, trainer, save_gguf: bool = False):
        """
        保存训练后的模型
        
        Args:
            trainer: 训练器对象
            save_gguf: 是否保存 GGUF 格式（Windows 上可能无法使用）
        """
        output_dir = self.config['training']['output_dir']
        
        # 1. 保存 LoRA 适配器
        final_model_path = f"{output_dir}/final_model"
        print(f"\n保存 LoRA 适配器到: {final_model_path}")
        trainer.save_model(final_model_path)
        print("✅ LoRA 适配器保存成功!")
        
        # 2. 保存合并后的 16-bit 完整模型
        merged_16bit_path = f"{output_dir}/merged_16bit"
        print(f"\n保存合并后的 16-bit 模型到: {merged_16bit_path}")
        try:
            self.model.save_pretrained_merged(
                merged_16bit_path,
                self.tokenizer,
                save_method="merged_16bit"
            )
            print("✅ 16-bit 合并模型保存成功!")
        except Exception as e:
            print(f"⚠️  16-bit 合并模型保存失败: {e}")
        
        # 3. 保存为 GGUF 格式（可选）
        if save_gguf:
            gguf_path = f"{output_dir}/gguf_model"
            print(f"\n保存为 GGUF 格式到: {gguf_path}")
            try:
                self.model.save_pretrained_gguf(
                    gguf_path,
                    self.tokenizer,
                    quantization_method="q4_k_m"
                )
                print("✅ GGUF 模型保存成功!")
            except Exception as e:
                print(f"⚠️  GGUF 保存失败（这是可选的）: {e}")
        
        print(f"\n{'='*60}")
        print(f"   所有模型已保存到: {output_dir}")
        print(f"{'='*60}")
    
    def get_training_stats(self, trainer):
        """获取训练统计信息"""
        if trainer.state.log_history:
            print("\n" + "="*60)
            print("   训练统计")
            print("="*60)
            
            # 查找最后的训练损失
            train_loss = None
            for log in reversed(trainer.state.log_history):
                if 'loss' in log:
                    train_loss = log['loss']
                    break
            
            if train_loss is not None:
                print(f"最终训练损失: {train_loss:.4f}")
            
            # 查找最后的验证损失
            if self.eval_dataset:
                eval_loss = None
                for log in reversed(trainer.state.log_history):
                    if 'eval_loss' in log:
                        eval_loss = log['eval_loss']
                        break
                
                if eval_loss is not None:
                    print(f"最终验证损失: {eval_loss:.4f}")
            
            print("="*60)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("   Figma JSON 微调训练")
    print("="*60)
    
    # 检查 CUDA 是否可用
    print(f"\nCUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*60 + "\n")
    
    # 创建训练器
    trainer_obj = FigmaJSONTrainer(config_path="config.yaml")
    
    # 加载模型
    trainer_obj.load_model()
    
    # 加载数据集
    trainer_obj.load_datasets()
    
    # 开始训练
    trainer = trainer_obj.train()
    
    # 显示统计信息
    trainer_obj.get_training_stats(trainer)
    
    print("\n" + "="*60)
    print("   ✅ 训练流程全部完成!")
    print("="*60)
    print(f"\n模型已保存到 ./outputs/ 目录")
    print("使用以下命令进行推理:")
    print("  python inference.py --model_path ./outputs/final_model --mode interactive")
    print()


if __name__ == "__main__":
    main()
