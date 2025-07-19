#!/usr/bin/env python3
"""
上传模型到Hugging Face Hub
"""

import os
import sys
import argparse
from loguru import logger

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def upload_to_huggingface(model_path: str, repo_name: str, token: str):
    """上传模型到Hugging Face Hub"""
    try:
        from huggingface_hub import HfApi, create_repo
        
        # 创建API客户端
        api = HfApi(token=token)
        
        # 创建仓库（如果不存在）
        try:
            create_repo(repo_name, token=token, exist_ok=True)
            logger.info(f"仓库 {repo_name} 创建成功或已存在")
        except Exception as e:
            logger.warning(f"创建仓库时出现警告: {e}")
        
        # 上传模型文件
        logger.info(f"开始上传模型到 {repo_name}...")
        
        # 上传所有文件
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            token=token,
            ignore_patterns=["*.log", "*.tmp", "__pycache__/*"]
        )
        
        logger.info(f"模型上传成功！访问地址: https://huggingface.co/{repo_name}")
        
    except ImportError:
        logger.error("请先安装 huggingface_hub: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"上传失败: {e}")
        return False
    
    return True


def create_model_card(repo_name: str, output_path: str = "model_card.md"):
    """创建模型卡片"""
    try:
        with open("model_card.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # 替换占位符
        content = content.replace("your-username", repo_name.split("/")[0])
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"模型卡片已创建: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"创建模型卡片失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="上传模型到Hugging Face Hub")
    parser.add_argument("--model_path", type=str, default="outputs",
                       help="模型路径")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="Hugging Face仓库名称 (格式: username/repo-name)")
    parser.add_argument("--token", type=str, required=True,
                       help="Hugging Face访问令牌")
    parser.add_argument("--create_card", action="store_true",
                       help="是否创建模型卡片")
    
    args = parser.parse_args()
    
    # 设置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return
    
    # 检查仓库名称格式
    if "/" not in args.repo_name:
        logger.error("仓库名称格式错误，应为: username/repo-name")
        return
    
    # 创建模型卡片
    if args.create_card:
        create_model_card(args.repo_name)
    
    # 上传模型
    success = upload_to_huggingface(args.model_path, args.repo_name, args.token)
    
    if success:
        logger.info("=" * 60)
        logger.info("🎉 模型上传完成！")
        logger.info(f"📁 仓库地址: https://huggingface.co/{args.repo_name}")
        logger.info("")
        logger.info("📋 下一步操作：")
        logger.info("1. 在Hugging Face上完善模型描述")
        logger.info("2. 添加模型标签和分类")
        logger.info("3. 上传模型卡片")
        logger.info("4. 测试模型功能")
        logger.info("=" * 60)
    else:
        logger.error("模型上传失败，请检查错误信息")


if __name__ == "__main__":
    main() 