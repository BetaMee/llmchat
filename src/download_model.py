#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 HuggingFace 下载模型到本地
支持自定义本地目录，便于离线使用
"""

import os
import argparse
from huggingface_hub import snapshot_download


# 可用的模型列表
AVAILABLE_MODELS = {
    "qwen-7b": {
        "model_id": "unsloth/Qwen2.5-7B-Instruct",
        "描述": "7B 参数，推荐用于 24GB 显存的 GPU",
        "显存需求": "~12-15GB"
    },
    "qwen-14b": {
        "model_id": "unsloth/Qwen2.5-14B-Instruct",
        "描述": "14B 参数，需要更大显存",
        "显存需求": "~22-26GB"
    },
    "qwen-3b": {
        "model_id": "unsloth/Qwen2.5-3B-Instruct",
        "描述": "3B 参数，显存友好的小模型",
        "显存需求": "~6-8GB"
    },
}


def list_models():
    """列出可用的模型"""
    print("\n" + "="*60)
    print("   可用模型列表")
    print("="*60 + "\n")
    
    for key, info in AVAILABLE_MODELS.items():
        print(f"🔹 {key}")
        print(f"   Model ID: {info['model_id']}")
        print(f"   描述: {info['描述']}")
        print(f"   显存需求: {info['显存需求']}")
        print()


def download_model(model_key: str, local_dir: str = "./models"):
    """
    下载模型到本地目录
    
    Args:
        model_key: 模型键名（如 'qwen-7b'）
        local_dir: 本地保存目录
    """
    if model_key not in AVAILABLE_MODELS:
        print(f"❌ 错误: 未知的模型 '{model_key}'")
        print(f"可用模型: {', '.join(AVAILABLE_MODELS.keys())}")
        print("使用 --list 查看所有模型")
        return False
    
    model_info = AVAILABLE_MODELS[model_key]
    model_id = model_info['model_id']
    
    # 构建本地保存路径
    model_name = model_id.split('/')[-1]
    save_path = os.path.join(local_dir, model_name)
    
    print("\n" + "="*60)
    print(f"   下载模型: {model_key}")
    print("="*60)
    print(f"\nModel ID: {model_id}")
    print(f"描述: {model_info['描述']}")
    print(f"显存需求: {model_info['显存需求']}")
    print(f"保存路径: {save_path}")
    print("\n开始下载（这可能需要一些时间）...\n")
    
    try:
        # 从 HuggingFace 下载到本地目录
        model_dir = snapshot_download(
            model_id,
            local_dir=save_path,
            local_dir_use_symlinks=False
        )
        
        print("\n" + "="*60)
        print("   ✅ 下载完成!")
        print("="*60)
        print(f"\n模型已保存到: {model_dir}")
        
        # 显示如何使用
        print("\n" + "="*60)
        print("   更新配置文件")
        print("="*60)
        print("\n请更新 config.yaml，使用以下配置:")
        print(f"\nmodel:")
        print(f'  name: "{model_dir}"')
        print(f"  max_seq_length: 4096")
        print(f"  dtype: null")
        print(f"  load_in_4bit: true")
        
        # 自动更新配置（可选）
        update = input("\n是否自动更新 config.yaml? (y/n): ").strip().lower()
        if update == 'y':
            update_config(model_dir)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接问题（可能需要代理）")
        print("  2. HuggingFace 服务异常")
        print("  3. 磁盘空间不足")
        print("\n提示: 如果需要使用镜像，请设置环境变量:")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
        return False


def update_config(model_path: str):
    """更新配置文件"""
    import yaml
    
    config_file = "config.yaml"
    
    try:
        # 读取现有配置
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 更新模型路径
        config['model']['name'] = model_path
        
        # 保存配置
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        print(f"\n✅ 配置文件已更新: {config_file}")
        print(f"   模型路径: {model_path}")
        
    except Exception as e:
        print(f"\n⚠️  自动更新配置失败: {e}")
        print("请手动更新 config.yaml")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载 Qwen 模型到本地",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有可用模型
  python src/download_model.py --list
  
  # 下载 Qwen 7B 模型（推荐）
  python src/download_model.py --model qwen-7b
  
  # 下载到指定目录
  python src/download_model.py --model qwen-7b --local-dir /path/to/models
  
  # 下载 14B 模型
  python src/download_model.py --model qwen-14b
  
  # 使用 HuggingFace 镜像（国内访问）
  export HF_ENDPOINT=https://hf-mirror.com
  python src/download_model.py --model qwen-7b
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的模型"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        help="要下载的模型"
    )
    
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./models",
        help="模型本地保存目录（默认: ./models）"
    )
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("\n" + "="*60)
    print("   HuggingFace 模型下载工具")
    print("="*60)
    
    if args.list:
        # 列出模型
        list_models()
    elif args.model:
        # 下载模型
        success = download_model(args.model, args.local_dir)
        if success:
            print("\n下一步:")
            print("  1. 准备训练数据")
            print("  2. 运行: ./start_train.sh --offline")
            print()
    else:
        # 交互式选择
        print("\n请选择要下载的模型:\n")
        for i, (key, info) in enumerate(AVAILABLE_MODELS.items(), 1):
            print(f"{i}. {key}")
            print(f"   {info['描述']}")
            print(f"   显存需求: {info['显存需求']}\n")
        
        try:
            choice = int(input("请输入序号 (1-{}): ".format(len(AVAILABLE_MODELS))))
            model_key = list(AVAILABLE_MODELS.keys())[choice - 1]
            download_model(model_key, args.local_dir)
        except (ValueError, IndexError):
            print("\n❌ 无效的选择")
        except KeyboardInterrupt:
            print("\n\n已取消")


if __name__ == "__main__":
    main()
