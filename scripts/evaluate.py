#!/usr/bin/env python3
"""
模型评估脚本
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from loguru import logger

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.inference import LogisticsExtractor


def load_test_data(filepath: str) -> List[Dict[str, Any]]:
    """加载测试数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculate_metrics(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算评估指标"""
    total = len(predictions)
    correct = 0
    partial_correct = 0
    
    for pred, gt in zip(predictions, ground_truth):
        if pred["success"]:
            pred_info = pred["extracted_info"]
            gt_info = json.loads(gt["output"])
            
            # 完全匹配
            if pred_info == gt_info:
                correct += 1
                partial_correct += 1
            else:
                # 部分匹配（检查关键字段）
                key_fields = ["收件人", "收件地址", "收件人电话"]
                matched_fields = 0
                total_fields = 0
                
                for field in key_fields:
                    if field in gt_info and field in pred_info:
                        total_fields += 1
                        if pred_info[field] == gt_info[field]:
                            matched_fields += 1
                
                if total_fields > 0 and matched_fields / total_fields >= 0.8:
                    partial_correct += 1
    
    accuracy = correct / total if total > 0 else 0
    partial_accuracy = partial_correct / total if total > 0 else 0
    
    return {
        "total_samples": total,
        "exact_match_accuracy": accuracy,
        "partial_match_accuracy": partial_accuracy,
        "success_rate": sum(1 for p in predictions if p["success"]) / total if total > 0 else 0
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估物流信息提取模型")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--test_file", type=str, default="data/test.json",
                       help="测试数据文件")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="评估结果输出文件")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备类型")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大测试样本数")
    
    args = parser.parse_args()
    
    # 设置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.test_file):
        logger.error(f"测试数据文件不存在: {args.test_file}")
        return
    
    # 加载测试数据
    logger.info(f"正在加载测试数据: {args.test_file}")
    test_data = load_test_data(args.test_file)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    logger.info(f"测试数据加载完成，共 {len(test_data)} 条")
    
    # 创建提取器
    logger.info("正在加载模型...")
    extractor = LogisticsExtractor(args.model_path, args.device)
    
    # 进行预测
    logger.info("开始评估...")
    predictions = []
    
    for i, sample in enumerate(test_data):
        if (i + 1) % 100 == 0:
            logger.info(f"已处理 {i + 1}/{len(test_data)} 条数据")
        
        result = extractor.extract_information(sample["input"])
        predictions.append(result)
    
    # 计算指标
    logger.info("正在计算评估指标...")
    metrics = calculate_metrics(predictions, test_data)
    
    # 显示结果
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"总样本数: {metrics['total_samples']}")
    print(f"完全匹配准确率: {metrics['exact_match_accuracy']:.4f}")
    print(f"部分匹配准确率: {metrics['partial_match_accuracy']:.4f}")
    print(f"成功率: {metrics['success_rate']:.4f}")
    print("="*50)
    
    # 保存结果
    evaluation_results = {
        "model_path": args.model_path,
        "test_file": args.test_file,
        "metrics": metrics,
        "predictions": predictions[:10]  # 只保存前10个预测结果作为示例
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存到: {args.output_file}")
    
    # 显示一些示例
    print("\n示例预测结果:")
    for i in range(min(3, len(predictions))):
        print(f"\n样本 {i+1}:")
        print(f"  输入: {test_data[i]['input']}")
        print(f"  真实: {test_data[i]['output']}")
        if predictions[i]["success"]:
            print(f"  预测: {json.dumps(predictions[i]['extracted_info'], ensure_ascii=False)}")
        else:
            print(f"  预测: 失败 - {predictions[i]['error']}")


if __name__ == "__main__":
    main() 