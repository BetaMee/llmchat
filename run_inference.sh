#!/bin/bash
# Figma JSON 推理启动脚本

echo "=========================================="
echo "  Figma JSON 模型推理"
echo "=========================================="

echo ""
echo "开始推理..."
echo ""

# 运行推理
# 根据需要修改参数
python src/inference.py --model_path ./outputs/final_model "$@"

echo ""
echo "=========================================="
echo "  推理完成!"
echo "=========================================="
