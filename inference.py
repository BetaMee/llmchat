#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本
用于使用训练好的模型生成 Figma JSON
"""

import os
import json
import yaml
import torch
from unsloth import FastLanguageModel


class FigmaJSONInference:
    """Figma JSON 推理器"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        print(f"正在从 {self.model_path} 加载模型...")
        
        model_config = self.config['model']
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=model_config['max_seq_length'],
            dtype=model_config['dtype'],
            load_in_4bit=model_config['load_in_4bit'],
        )
        
        # 切换到推理模式
        FastLanguageModel.for_inference(self.model)
        
        print("模型加载完成!")
    
    def create_prompt(self, user_input: str) -> str:
        """
        创建推理 prompt
        
        Args:
            user_input: 用户输入的设计需求
            
        Returns:
            格式化的 prompt
        """
        prompt = f"""<|im_start|>system
你是一个专业的 Figma JSON 生成助手，能够根据用户的设计需求生成对应的 Figma JSON 格式数据。<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate(self, user_input: str, **kwargs) -> str:
        """
        生成 Figma JSON
        
        Args:
            user_input: 用户输入的设计需求
            **kwargs: 生成参数（会覆盖配置文件中的默认值）
            
        Returns:
            生成的 Figma JSON 字符串
        """
        # 获取生成配置
        inference_config = self.config['inference']
        max_new_tokens = kwargs.get('max_new_tokens', inference_config['max_new_tokens'])
        temperature = kwargs.get('temperature', inference_config['temperature'])
        top_p = kwargs.get('top_p', inference_config['top_p'])
        do_sample = kwargs.get('do_sample', inference_config['do_sample'])
        
        # 创建 prompt
        prompt = self.create_prompt(user_input)
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_seq_length']
        ).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手的回复
        if "<|im_start|>assistant" in full_output:
            assistant_output = full_output.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in assistant_output:
                assistant_output = assistant_output.split("<|im_end|>")[0].strip()
            return assistant_output
        
        return full_output
    
    def generate_and_parse(self, user_input: str, **kwargs) -> dict:
        """
        生成 Figma JSON 并解析为字典
        
        Args:
            user_input: 用户输入的设计需求
            **kwargs: 生成参数
            
        Returns:
            解析后的 JSON 字典
        """
        output = self.generate(user_input, **kwargs)
        
        try:
            # 尝试解析 JSON
            json_data = json.loads(output)
            return json_data
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            print(f"原始输出: {output}")
            return {"error": "JSON parse failed", "raw_output": output}
    
    def batch_generate(self, inputs: list, **kwargs) -> list:
        """
        批量生成
        
        Args:
            inputs: 用户输入列表
            **kwargs: 生成参数
            
        Returns:
            生成结果列表
        """
        results = []
        for i, user_input in enumerate(inputs):
            print(f"\n处理 {i+1}/{len(inputs)}: {user_input[:50]}...")
            result = self.generate(user_input, **kwargs)
            results.append(result)
        
        return results


def interactive_mode(inference: FigmaJSONInference):
    """交互式模式"""
    print("\n=== Figma JSON 生成器 - 交互模式 ===")
    print("输入设计需求，模型将生成对应的 Figma JSON")
    print("输入 'quit' 或 'exit' 退出\n")
    
    while True:
        user_input = input("请输入设计需求: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("退出交互模式")
            break
        
        if not user_input:
            continue
        
        print("\n生成中...")
        try:
            output = inference.generate(user_input)
            print("\n=== 生成结果 ===")
            print(output)
            
            # 尝试格式化 JSON
            try:
                json_obj = json.loads(output)
                print("\n=== 格式化的 JSON ===")
                print(json.dumps(json_obj, ensure_ascii=False, indent=2))
            except:
                pass
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"生成失败: {e}\n")


def test_examples(inference: FigmaJSONInference):
    """测试示例"""
    examples = [
        "创建一个蓝色的矩形按钮，宽度200px，高度50px",
        "生成一个红色圆形，半径30px",
        "创建一个文本框，内容为'Hello World'，字体大小24px",
        "创建一个绿色的卡片，宽度300px，高度200px，圆角12px",
    ]
    
    print("\n=== 测试示例 ===\n")
    
    for i, example in enumerate(examples, 1):
        print(f"\n[{i}] 输入: {example}")
        print("-" * 60)
        
        try:
            output = inference.generate(example)
            print("输出:")
            print(output)
            
            # 尝试解析和格式化
            try:
                json_obj = json.loads(output)
                print("\n格式化的 JSON:")
                print(json.dumps(json_obj, ensure_ascii=False, indent=2))
            except:
                pass
            
        except Exception as e:
            print(f"生成失败: {e}")
        
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Figma JSON 生成器")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/final_model",
        help="模型路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['interactive', 'test', 'single'],
        default='interactive',
        help="运行模式: interactive(交互), test(测试), single(单次)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="单次生成时的输入文本"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        print("请先运行训练脚本或指定正确的模型路径")
        return
    
    # 初始化推理器
    inference = FigmaJSONInference(model_path=args.model_path)
    
    # 根据模式运行
    if args.mode == 'interactive':
        interactive_mode(inference)
    
    elif args.mode == 'test':
        test_examples(inference)
    
    elif args.mode == 'single':
        if not args.input:
            print("错误: 单次模式需要提供 --input 参数")
            return
        
        print(f"输入: {args.input}")
        result = inference.generate(args.input)
        print(f"\n输出:\n{result}")
        
        # 保存到文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
