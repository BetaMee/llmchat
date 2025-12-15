#!/bin/bash
# Figma JSON 训练启动脚本

echo "=========================================="
echo "  Figma JSON 微调训练启动"
echo "=========================================="

# 配置 ModelScope 缓存目录
export MODELSCOPE_CACHE=./models
echo "✅ ModelScope 缓存目录: $MODELSCOPE_CACHE"

# 如果需要使用 HuggingFace 镜像（ModelScope 失败时的备选）
# export HF_ENDPOINT=https://hf-mirror.com

# 可选：完全离线模式（需要提前下载好模型）
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo ""
echo "环境检查:"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "  开始训练..."
echo "=========================================="
echo ""

# 启动训练
python train.py

echo ""
echo "=========================================="
echo "  训练完成!"
echo "=========================================="
