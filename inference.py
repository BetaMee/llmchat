#!/usr/bin/env python3
"""
物流填单信息提取推理脚本
"""

import os
import sys
import argparse
import json
from loguru import logger

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.inference import LogisticsExtractor


def setup_logging():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def interactive_mode(extractor):
    """交互模式"""
    print("欢迎使用物流信息提取器！")
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n请输入物流填单信息: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            # 提取信息
            result = extractor.extract_information(user_input)
            
            # 显示结果
            print("\n提取结果:")
            if result["success"]:
                print(json.dumps(result["extracted_info"], ensure_ascii=False, indent=2))
                
                # 验证结果
                validation = extractor.validate_extraction(result["extracted_info"])
                if not validation["is_valid"]:
                    print("\n验证警告:")
                    if validation["missing_fields"]:
                        print(f"  缺失字段: {', '.join(validation['missing_fields'])}")
                    if validation["invalid_fields"]:
                        print(f"  无效字段: {', '.join(validation['invalid_fields'])}")
            else:
                print(f"提取失败: {result['error']}")
                print(f"原始回复: {result['raw_response']}")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="物流填单信息提取推理")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--input_text", type=str,
                       help="输入文本（如果不提供则进入交互模式）")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备类型")
    parser.add_argument("--output_file", type=str,
                       help="输出文件路径（可选）")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return
    
    # 创建提取器
    logger.info("正在加载模型...")
    extractor = LogisticsExtractor(args.model_path, args.device)
    
    # 单次推理模式
    if args.input_text:
        logger.info("开始提取信息...")
        result = extractor.extract_information(args.input_text)
        
        # 显示结果
        print("输入文本:", args.input_text)
        print("提取结果:", json.dumps(result, ensure_ascii=False, indent=2))
        
        # 验证结果
        if result["success"]:
            validation = extractor.validate_extraction(result["extracted_info"])
            print("验证结果:", json.dumps(validation, ensure_ascii=False, indent=2))
        
        # 保存结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {args.output_file}")
    
    # 交互模式
    else:
        interactive_mode(extractor)


if __name__ == "__main__":
    main() 