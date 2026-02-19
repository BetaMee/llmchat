·#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figma JSON 数据处理脚本
用于准备训练数据
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path


class FigmaJSONProcessor:
    """处理 Figma JSON 数据的类"""
    
    def __init__(self, max_length: int = 4096):
        self.max_length = max_length
        
    def create_prompt(self, instruction: str, output_text: str) -> str:
        """
        创建训练 prompt
        
        Args:
            instruction: 用户的设计意图描述
            output_text: 对应的 Figma JSON 输出
            
        Returns:
            格式化的 prompt
        """
        prompt = f"""<|im_start|>system
你是一个专业的 Figma JSON 生成助手，能够根据用户的设计意图描述生成对应的 Figma 节点 JSON 格式数据。请严格按照 Figma API 规范生成 JSON，确保包含所有必要字段如 type、name、width、height、children 等。<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""
        return prompt
    
    def process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本
        
        Args:
            sample: 包含 'instruction' 和 'output' 键的字典
            
        Returns:
            处理后的样本
        """
        instruction = sample.get('instruction', '')
        output_text = sample.get('output', '')
        
        # 如果 output 是字典，转换为 JSON 字符串
        if isinstance(output_text, dict):
            output_text = json.dumps(output_text, ensure_ascii=False, separators=(',', ':'))
        
        prompt = self.create_prompt(instruction, output_text)
        
        return {
            'text': prompt,
            'instruction': instruction,
            'output': output_text
        }
    
    def process_dataset(self, input_file: str, output_file: str):
        """
        处理整个数据集
        
        Args:
            input_file: 输入文件路径（JSONL格式）
            output_file: 输出文件路径（JSON格式）
        """
        # 读取原始数据（支持 JSONL 和 JSON 格式）
        raw_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 尝试作为 JSON 数组读取
            try:
                raw_data = json.loads(content)
            except json.JSONDecodeError:
                # 作为 JSONL 逐行读取
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            raw_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"解析行时出错: {e}")
                            continue
        
        print(f"成功读取 {len(raw_data)} 条原始数据")
        
        # 处理每个样本
        processed_data = []
        skipped = 0
        for i, sample in enumerate(raw_data):
            try:
                # 检查必需字段
                if 'instruction' not in sample or 'output' not in sample:
                    print(f"样本 {i} 缺少必需字段，跳过")
                    skipped += 1
                    continue
                
                processed_sample = self.process_single_sample(sample)
                processed_data.append(processed_sample)
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                skipped += 1
                continue
        
        # 保存处理后的数据
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成!")
        print(f"  有效样本: {len(processed_data)}")
        print(f"  跳过样本: {skipped}")
        print(f"  保存至: {output_file}")
        
        # 显示第一个样本示例
        if processed_data:
            print(f"\n第一个样本预览:")
            print(f"  指令: {processed_data[0]['instruction'][:100]}...")
            print(f"  输出预览: {processed_data[0]['output'][:150]}...")
    
    def create_sample_data(self, output_dir: str = "./data"):
        """
        创建示例数据（用于测试）
        
        Args:
            output_dir: 输出目录
        """
        sample_data = [
            {
                "instruction": "创建一个蓝色的矩形按钮，宽度200px，高度50px，圆角8px",
                "input": "",
                "output": {
                    "type": "RECTANGLE",
                    "name": "Button",
                    "width": 200,
                    "height": 50,
                    "x": 0,
                    "y": 0,
                    "fills": [
                        {
                            "type": "SOLID",
                            "color": {"r": 0.0, "g": 0.5, "b": 1.0, "a": 1.0}
                        }
                    ],
                    "cornerRadius": 8,
                    "blendMode": "PASS_THROUGH"
                }
            },
            {
                "instruction": "生成一个红色圆形，直径60px",
                "input": "",
                "output": {
                    "type": "ELLIPSE",
                    "name": "Circle",
                    "width": 60,
                    "height": 60,
                    "x": 0,
                    "y": 0,
                    "fills": [
                        {
                            "type": "SOLID",
                            "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
                        }
                    ],
                    "blendMode": "PASS_THROUGH"
                }
            },
            {
                "instruction": "创建一个文本框，内容为'Hello World'，字体大小24px，Inter Medium字体",
                "input": "",
                "output": {
                    "type": "TEXT",
                    "name": "Text",
                    "width": 150,
                    "height": 30,
                    "x": 0,
                    "y": 0,
                    "characters": "Hello World",
                    "fontSize": 24,
                    "fontName": {
                        "family": "Inter",
                        "style": "Medium"
                    },
                    "fills": [
                        {
                            "type": "SOLID",
                            "color": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}
                        }
                    ],
                    "textAlignHorizontal": "LEFT",
                    "textAlignVertical": "TOP",
                    "blendMode": "PASS_THROUGH"
                }
            }
        ]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建训练集 (80%)
        train_size = int(len(sample_data) * 0.8)
        train_data = sample_data[:train_size] if train_size > 0 else sample_data[:1]
        
        # 创建验证集 (20%)
        val_data = sample_data[train_size:] if train_size < len(sample_data) else [sample_data[0]]
        
        # 保存为 JSONL 格式
        with open(f"{output_dir}/raw_train.jsonl", 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(f"{output_dir}/raw_val.jsonl", 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"示例数据已创建在 {output_dir} 目录")
        print(f"  训练集: {len(train_data)} 条")
        print(f"  验证集: {len(val_data)} 条")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Figma JSON 数据处理工具")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['sample', 'process'],
        default='sample',
        help="运行模式: sample(创建示例), process(处理数据)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="输入文件路径（JSONL或JSON格式）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    processor = FigmaJSONProcessor()
    data_dir = "./data"
    
    if args.mode == 'sample':
        # 创建示例数据
        print("创建示例数据...")
        processor.create_sample_data(data_dir)
        
        # 处理示例数据
        print("\n处理训练数据...")
        processor.process_dataset(
            f"{data_dir}/raw_train.jsonl",
            f"{data_dir}/train.json"
        )
        
        print("\n处理验证数据...")
        processor.process_dataset(
            f"{data_dir}/raw_val.jsonl",
            f"{data_dir}/val.json"
        )
        
    elif args.mode == 'process':
        if not args.input or not args.output:
            print("错误: process 模式需要提供 --input 和 --output 参数")
            return
        
        print(f"处理数据: {args.input} -> {args.output}")
        processor.process_dataset(args.input, args.output)
    
    print("\n数据处理完成!")


if __name__ == "__main__":
    main()
