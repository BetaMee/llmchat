#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figma JSON 数据处理脚本
用于准备训练数据
"""

import json
import os
from typing import List, Dict
from pathlib import Path


class FigmaJSONProcessor:
    """处理 Figma JSON 数据的类"""
    
    def __init__(self, max_length: int = 2048):
        self.max_length = max_length
        
    def create_prompt(self, input_text: str, output_text: str) -> str:
        """
        创建训练 prompt
        
        Args:
            input_text: 输入的 Figma 描述或需求
            output_text: 对应的 Figma JSON 输出
            
        Returns:
            格式化的 prompt
        """
        prompt = f"""<|im_start|>system
你是一个专业的 Figma JSON 生成助手，能够根据用户的设计需求生成对应的 Figma JSON 格式数据。<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""
        return prompt
    
    def process_single_sample(self, sample: Dict) -> Dict:
        """
        处理单个样本
        
        Args:
            sample: 包含 'input' 和 'output' 键的字典
            
        Returns:
            处理后的样本
        """
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')
        
        # 如果 output 是字典，转换为 JSON 字符串
        if isinstance(output_text, dict):
            output_text = json.dumps(output_text, ensure_ascii=False, indent=2)
        
        prompt = self.create_prompt(input_text, output_text)
        
        return {
            'text': prompt,
            'input': input_text,
            'output': output_text
        }
    
    def process_dataset(self, input_file: str, output_file: str):
        """
        处理整个数据集
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 处理每个样本
        processed_data = []
        for sample in raw_data:
            try:
                processed_sample = self.process_single_sample(sample)
                processed_data.append(processed_sample)
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue
        
        # 保存处理后的数据
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成! 共处理 {len(processed_data)} 个样本")
        print(f"保存至: {output_file}")
    
    def create_sample_data(self, output_dir: str = "./data"):
        """
        创建示例数据
        
        Args:
            output_dir: 输出目录
        """
        sample_data = [
            {
                "input": "创建一个蓝色的矩形按钮，宽度200px，高度50px",
                "output": {
                    "type": "RECTANGLE",
                    "name": "Button",
                    "width": 200,
                    "height": 50,
                    "fills": [
                        {
                            "type": "SOLID",
                            "color": {"r": 0.0, "g": 0.5, "b": 1.0, "a": 1.0}
                        }
                    ],
                    "cornerRadius": 8
                }
            },
            {
                "input": "生成一个红色圆形，半径30px",
                "output": {
                    "type": "ELLIPSE",
                    "name": "Circle",
                    "width": 60,
                    "height": 60,
                    "fills": [
                        {
                            "type": "SOLID",
                            "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
                        }
                    ]
                }
            },
            {
                "input": "创建一个文本框，内容为'Hello World'，字体大小24px",
                "output": {
                    "type": "TEXT",
                    "name": "Text",
                    "characters": "Hello World",
                    "fontSize": 24,
                    "fontName": {
                        "family": "Inter",
                        "style": "Regular"
                    },
                    "fills": [
                        {
                            "type": "SOLID",
                            "color": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}
                        }
                    ]
                }
            }
        ]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建训练集 (80%)
        train_size = int(len(sample_data) * 0.8)
        train_data = sample_data[:train_size] * 10  # 复制数据以增加样本量
        
        # 创建验证集和测试集
        val_data = sample_data[train_size:]
        test_data = sample_data[:1]
        
        # 保存数据
        with open(f"{output_dir}/raw_train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(f"{output_dir}/raw_val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        with open(f"{output_dir}/raw_test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"示例数据已创建在 {output_dir} 目录")


def main():
    """主函数"""
    processor = FigmaJSONProcessor()
    
    # 创建数据目录
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建示例数据
    print("创建示例数据...")
    processor.create_sample_data(data_dir)
    
    # 处理数据
    print("\n处理训练数据...")
    processor.process_dataset(
        f"{data_dir}/raw_train.json",
        f"{data_dir}/train.json"
    )
    
    print("\n处理验证数据...")
    processor.process_dataset(
        f"{data_dir}/raw_val.json",
        f"{data_dir}/val.json"
    )
    
    print("\n处理测试数据...")
    processor.process_dataset(
        f"{data_dir}/raw_test.json",
        f"{data_dir}/test.json"
    )
    
    print("\n数据处理完成!")


if __name__ == "__main__":
    main()
