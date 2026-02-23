#!/bin/bash
# vLLM 服务启动脚本

echo "=========================================="
echo "  Figma JSON vLLM 服务"
echo "=========================================="

# 默认参数
MODEL_PATH="${1:-./outputs/final_model}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"

echo ""
echo "模型路径: $MODEL_PATH"
echo "服务地址: http://$HOST:$PORT"
echo ""

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    echo "请先训练模型或指定正确的路径"
    exit 1
fi

# 启动服务
echo "启动 vLLM 服务..."
echo ""

python src/vllm_server.py \
    --model_path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --max_seq_length 2048
