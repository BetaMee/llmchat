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
        print(f"  模型: {self.model_path}")
        print(f"  最大序列长度: {model_config['max_seq_length']}")
    
    def create_prompt(self, instruction: str) -> str:
        """
        创建推理 prompt
        
        Args:
            instruction: 用户输入的设计意图
            
        Returns:
            格式化的 prompt
        """
        prompt = f"""<|im_start|>system
你是一个专业的 Figma JSON 生成助手，能够根据用户的设计意图描述生成对应的 Figma 节点 JSON 格式数据。请严格按照 Figma API 规范生成 JSON，确保包含所有必要字段如 type、name、width、height、children 等。<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate(self, instruction: str, **kwargs) -> str:
        """
        生成 Figma JSON
        
        Args:
            instruction: 用户输入的设计意图
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
        prompt = self.create_prompt(instruction)
        
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
    
    def generate_and_parse(self, instruction: str, **kwargs) -> dict:
        """
        生成 Figma JSON 并解析为字典
        
        Args:
            instruction: 用户输入的设计意图
            **kwargs: 生成参数
            
        Returns:
            解析后的 JSON 字典
        """
        output = self.generate(instruction, **kwargs)
        
        try:
            # 尝试解析 JSON
            json_data = json.loads(output)
            return json_data
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            print(f"原始输出: {output}")
            return {"error": "JSON parse failed", "raw_output": output}
    
    def batch_generate(self, instructions: list, **kwargs) -> list:
        """
        批量生成
        
        Args:
            instructions: 用户输入列表
            **kwargs: 生成参数
            
        Returns:
            生成结果列表
        """
        results = []
        for i, instruction in enumerate(instructions):
            print(f"\n处理 {i+1}/{len(instructions)}: {instruction[:50]}...")
            result = self.generate(instruction, **kwargs)
            results.append(result)
        
        return results


def interactive_mode(inference: FigmaJSONInference):
    """交互式模式"""
    print("\n" + "="*60)
    print("   Figma JSON 生成器 - 交互模式")
    print("="*60)
    print("输入设计意图，模型将生成对应的 Figma JSON")
    print("输入 'quit' 或 'exit' 或 'q' 退出")
    print("="*60 + "\n")
    
    while True:
        instruction = input("💡 请输入设计意图: ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("\n👋 退出交互模式")
            break
        
        if not instruction:
            continue
        
        print("\n⏳ 生成中...")
        try:
            output = inference.generate(instruction)
            print("\n" + "="*60)
            print("📝 生成结果:")
            print("="*60)
            print(output)
            
            # 尝试格式化 JSON
            try:
                json_obj = json.loads(output)
                print("\n" + "="*60)
                print("✨ 格式化的 JSON:")
                print("="*60)
                print(json.dumps(json_obj, ensure_ascii=False, indent=2))
            except Exception as parse_error:
                print(f"\n⚠️  JSON 解析警告: {parse_error}")
            
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ 生成失败: {e}\n")


def test_examples(inference: FigmaJSONInference):
    """测试示例"""
    examples = [
        "创建一个蓝色的矩形按钮，宽度200px，高度50px，圆角8px",
        "生成一个红色圆形，直径60px",
        "创建一个文本框，内容为'Hello World'，字体大小24px，Inter Medium字体",
        "简约扁平风格UI预览界面，三列等宽圆角矩形卡片，浅灰填充，深灰背景",
    ]
    
    print("\n" + "="*60)
    print("   测试示例")
    print("="*60 + "\n")
    
    for i, example in enumerate(examples, 1):
        print(f"\n[测试 {i}/{len(examples)}]")
        print(f"💡 输入: {example}")
        print("-" * 60)
        
        try:
            output = inference.generate(example)
            print("📝 输出:")
            print(output)
            
            # 尝试解析和格式化
            try:
                json_obj = json.loads(output)
                print("\n✨ 格式化的 JSON:")
                print(json.dumps(json_obj, ensure_ascii=False, indent=2))
            except Exception:
                pass
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
        
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
    parser.add_argument(
        "--temperature",
        type=float,
        help="生成温度（覆盖配置文件）"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="最大生成token数（覆盖配置文件）"
    )
    
    args = parser.parse_args()
    
    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型路径不存在: {args.model_path}")
        print("请先运行训练脚本或指定正确的模型路径")
        return
    
    # 初始化推理器
    inference = FigmaJSONInference(model_path=args.model_path)
    
    # 准备生成参数
    gen_kwargs = {}
    if args.temperature is not None:
        gen_kwargs['temperature'] = args.temperature
    if args.max_new_tokens is not None:
        gen_kwargs['max_new_tokens'] = args.max_new_tokens
    
    # 根据模式运行
    if args.mode == 'interactive':
        interactive_mode(inference)
    
    elif args.mode == 'test':
        test_examples(inference)
    
    elif args.mode == 'single':
        if not args.input:
            print("❌ 错误: 单次模式需要提供 --input 参数")
            return
        
        print(f"💡 输入: {args.input}")
        result = inference.generate(args.input, **gen_kwargs)
        print(f"\n📝 输出:\n{result}")
        
        # 保存到文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✅ 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
