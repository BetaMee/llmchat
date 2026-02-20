#!/bin/bash
# Figma JSON 训练启动脚本
# 用法:
#   ./start_train.sh           # 在线模式（自动下载模型，源网速度可能会慢）
#   ./start_train.sh --offline # 离线模式（使用本地缓存）
#   ./start_train.sh --mirror  # 使用 HuggingFace 镜像

# 解析参数
OFFLINE_MODE=false
USE_MIRROR=false

for arg in "$@"; do
    case $arg in
        --offline)
            OFFLINE_MODE=true
            shift
            ;;
        --mirror)
            USE_MIRROR=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --offline    离线模式（使用本地缓存的模型）"
            echo "  --mirror     使用 HuggingFace 镜像站（hf-mirror.com）"
            echo "  --help, -h   显示此帮助信息"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "  Figma JSON 微调训练启动"
echo "=========================================="

# HuggingFace 配置
if [ "$OFFLINE_MODE" = true ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "✅ 离线模式已启用（使用本地缓存）"
else
    echo "✅ 在线模式已启用（自动下载模型）"
fi

# HuggingFace 镜像
if [ "$USE_MIRROR" = true ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo "✅ HuggingFace 镜像已启用: hf-mirror.com"
fi

echo ""
echo "环境检查:"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "  开始训练..."
echo "=========================================="
echo ""

# 启动训练
python src/train.py

echo ""
echo "=========================================="
echo "  训练完成!"
echo "=========================================="
