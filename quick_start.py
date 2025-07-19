#!/usr/bin/env python3
"""
物流填单信息提取模型 - 快速开始脚本
"""

import os
import sys
import subprocess
from loguru import logger


def run_command(command: str, description: str):
    """运行命令"""
    logger.info(f"开始{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"{description}完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description}失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False


def main():
    """主函数"""
    logger.info("欢迎使用物流填单信息提取模型！")
    logger.info("=" * 60)
    
    # 检查Python环境
    logger.info("检查Python环境...")
    if sys.version_info < (3, 8):
        logger.error("需要Python 3.8或更高版本")
        return
    
    # 安装依赖
    logger.info("安装依赖包...")
    if not run_command("pip install -r requirements.txt", "安装依赖"):
        return
    
    # 生成训练数据
    logger.info("生成训练数据...")
    if not run_command("python scripts/generate_data.py", "生成训练数据"):
        return
    
    # 开始训练
    logger.info("开始训练模型...")
    if not run_command("python train.py", "训练模型"):
        return
    
    # 评估模型
    logger.info("评估模型性能...")
    if not run_command("python scripts/evaluate.py --model_path outputs", "评估模型"):
        return
    
    # 测试推理
    logger.info("测试模型推理...")
    test_text = "收件人：张三，收件地址：北京市朝阳区中关村大街123号，收件人电话：13800138000"
    test_command = f'python inference.py --model_path outputs --input_text "{test_text}"'
    if not run_command(test_command, "测试推理"):
        return
    
    logger.info("=" * 60)
    logger.info("🎉 恭喜！模型训练和测试完成！")
    logger.info("")
    logger.info("📁 项目文件说明：")
    logger.info("  - outputs/          : 训练好的模型")
    logger.info("  - data/             : 训练数据")
    logger.info("  - configs/          : 配置文件")
    logger.info("  - src/              : 源代码")
    logger.info("")
    logger.info("🚀 使用方法：")
    logger.info("  1. 交互式推理: python inference.py --model_path outputs")
    logger.info("  2. 批量推理: python inference.py --model_path outputs --input_text '你的文本'")
    logger.info("  3. 模型评估: python scripts/evaluate.py --model_path outputs")
    logger.info("")
    logger.info("📊 下一步：")
    logger.info("  1. 在Hugging Face上发布模型")
    logger.info("  2. 优化模型性能")
    logger.info("  3. 扩展数据集")
    logger.info("=" * 60)


if __name__ == "__main__":
    main() 