#!/usr/bin/env python3
"""
数据生成脚本
"""

import os
import sys
import argparse
from loguru import logger

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_processing.data_generator import LogisticsDataGenerator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成物流填单训练数据")
    parser.add_argument("--train_samples", type=int, default=5000,
                       help="训练样本数量")
    parser.add_argument("--eval_samples", type=int, default=500,
                       help="验证样本数量")
    parser.add_argument("--test_samples", type=int, default=500,
                       help="测试样本数量")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 设置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据生成器
    generator = LogisticsDataGenerator()
    
    # 生成训练数据
    logger.info(f"开始生成训练数据 ({args.train_samples} 条)...")
    train_data = generator.generate_dataset(args.train_samples)
    generator.save_dataset(train_data, os.path.join(args.output_dir, "train.json"))
    
    # 生成验证数据
    logger.info(f"开始生成验证数据 ({args.eval_samples} 条)...")
    eval_data = generator.generate_dataset(args.eval_samples)
    generator.save_dataset(eval_data, os.path.join(args.output_dir, "eval.json"))
    
    # 生成测试数据
    logger.info(f"开始生成测试数据 ({args.test_samples} 条)...")
    test_data = generator.generate_dataset(args.test_samples)
    generator.save_dataset(test_data, os.path.join(args.output_dir, "test.json"))
    
    # 打印示例
    logger.info("数据生成完成！示例数据：")
    for i in range(3):
        sample = generator.generate_logistics_form()
        print(f"\n样本 {i+1}:")
        print(f"  输入: {sample['input']}")
        print(f"  输出: {sample['output']}")
    
    logger.info(f"所有数据已保存到 {args.output_dir} 目录")


if __name__ == "__main__":
    main() 