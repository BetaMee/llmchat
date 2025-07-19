#!/usr/bin/env python3
"""
物流填单信息提取模型训练脚本
"""

import os
import sys
import argparse
from loguru import logger

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.trainer import LogisticsTrainer


def setup_logging():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="物流填单信息提取模型训练")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--generate_data", action="store_true",
                       help="是否生成训练数据")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 生成训练数据
    if args.generate_data:
        logger.info("开始生成训练数据...")
        from src.data_processing.data_generator import main as generate_data
        generate_data()
        logger.info("训练数据生成完成")
    
    # 检查数据文件是否存在
    if not os.path.exists("data/train.json"):
        logger.error("训练数据文件不存在，请先运行 --generate_data 生成数据")
        return
    
    # 创建训练器并开始训练
    logger.info("开始训练流程...")
    trainer = LogisticsTrainer(args.config)
    trainer.prepare_training()
    trainer.train()
    trainer.save_model_info()
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main() 