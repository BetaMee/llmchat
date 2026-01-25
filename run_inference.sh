#!/bin/bash
# Figma JSON 推理启动脚本

echo "=========================================="
echo "  Figma JSON 模型推理"
echo "=========================================="

# 🔥 重要：禁用 HuggingFace 在线检查（使用本地模型）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "✅ 离线模式已启用"

echo ""
echo "开始推理..."
echo ""

# 运行推理
# 根据需要修改参数
python inference.py --model_path ./outputs/final_model "$@"

echo ""
echo "=========================================="
echo "  推理完成!"
echo "=========================================="
