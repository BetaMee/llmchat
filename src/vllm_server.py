#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 服务部署脚本
提供 OpenAI 兼容的 API 接口
"""

import os
import sys
import json
import argparse
import time
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid


# ============ 数据模型 ============

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "figma-json-model"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None


class GenerateRequest(BaseModel):
    instruction: str
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048


# ============ 环境配置 ============

# 禁用 Triton 的 C 编译（避免需要系统 C 编译器）
os.environ["TRITON_BUILD_IMPL"] = "pytorch"
# 禁用 vLLM 的 torch.compile（避免 Triton 编译）
os.environ["VLLM_TORCH_COMPILE"] = "0"

# ============ 全局变量 ============

llm_engine: Optional[LLM] = None
system_prompt = """你是一个专业的 Figma JSON 生成助手，能够根据用户的设计意图描述生成对应的 Figma 节点 JSON 格式数据。请严格按照 Figma API 规范生成 JSON，确保包含所有必要字段如 type、name、width、height、children 等。"""


# ============ 辅助函数 ============

def create_chat_prompt(messages: list[ChatMessage]) -> str:
    """创建聊天格式的 prompt"""
    prompt_parts = []
    
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif msg.role == "user":
            prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif msg.role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    
    # 添加 assistant 开始标记
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)


def create_figma_prompt(instruction: str) -> str:
    """创建 Figma JSON 生成的 prompt"""
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""


def parse_output(output: str) -> str:
    """解析模型输出，提取 assistant 部分"""
    if "<|im_start|>assistant" in output:
        assistant_output = output.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in assistant_output:
            assistant_output = assistant_output.split("<|im_end|>")[0].strip()
        return assistant_output
    return output


# ============ FastAPI 应用 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global llm_engine
    
    # 启动时加载模型
    print("=" * 60)
    print("   正在加载 vLLM 模型...")
    print("=" * 60)
    
    model_path = app.state.model_path
    max_seq_length = app.state.max_seq_length
    
    # 检查是否是 LoRA 适配器路径（包含 adapter_config.json）
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    # 显存优化配置（针对 RTX 4070 12GB）
    gpu_memory_utilization = 0.80  # 使用 80% 显存，留更多余量
    max_num_seqs = 8  # 降低并发，避免爆显存
    
    if os.path.exists(adapter_config_path):
        # 是 LoRA 适配器，需要加载基础模型
        print(f"检测到 LoRA 适配器: {model_path}")
        print("正在读取基础模型信息...")
        
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model = adapter_config.get('base_model_name_or_path', 'unsloth/Qwen3-4B')
        print(f"基础模型: {base_model}")
        
        llm_engine = LLM(
            model=base_model,
            max_model_len=max_seq_length,
            dtype="bfloat16",
            load_format="auto",
            trust_remote_code=True,
            # 显存优化
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            # 禁用前缀缓存节省显存
            enable_prefix_caching=False,
            # 禁用 CUDA graph 节省显存
            enforce_eager=True,
        )
        
        # 加载 LoRA 适配器
        print(f"加载 LoRA 适配器...")
        from vllm.lora.request import LoRARequest
        app.state.lora_request = LoRARequest(
            "figma_adapter",
            1,
            model_path
        )
        print(f"LoRA 适配器加载完成!")
    else:
        # 是完整模型路径
        llm_engine = LLM(
            model=model_path,
            max_model_len=max_seq_length,
            dtype="bfloat16",
            quantization="bitsandbytes" if app.state.load_in_8bit else None,
            load_format="auto",
            trust_remote_code=True,
            # 显存优化
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=False,
            enforce_eager=True,
        )
    
    print("✅ 模型加载完成!")
    print(f"   模型路径: {model_path}")
    print(f"   最大序列长度: {max_seq_length}")
    print("=" * 60)
    
    # 保存模型名称供 /v1/models 使用
    app.state.model_name = "figma-json-model"
    
    # 设置 API Key（从环境变量读取，默认不启用鉴权）
    app.state.api_key = os.environ.get("API_KEY", None)
    if app.state.api_key:
        print(f"   API 鉴权已启用")
    
    yield
    
    # 关闭时清理
    print("\n正在关闭服务...")


app = FastAPI(
    title="Figma JSON vLLM API",
    description="基于 vLLM 的 Figma JSON 生成服务",
    version="1.0.0",
    lifespan=lifespan
)


# ============ 鉴权中间件 ============

async def verify_api_key(request: Request):
    """验证 API Key"""
    api_key = getattr(app.state, 'api_key', None)
    if api_key is None:
        return True  # 未设置 API Key，跳过鉴权
    
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Missing or invalid Authorization header")
    
    provided_key = auth_header.replace("Bearer ", "")
    if provided_key != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return True


# ============ API 路由 ============

@app.get("/")
async def root():
    """根路径 - 服务状态"""
    return {
        "status": "running",
        "service": "Figma JSON vLLM API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "generate": "/v1/generate",
            "models": "/v1/models",
            "status": "/v1/status",
            "clear_cache": "POST /v1/clear_cache",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "model_loaded": llm_engine is not None}


@app.get("/v1/status")
async def status():
    """服务状态与显存监控"""
    import torch
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}",
            "allocated_memory_gb": f"{torch.cuda.memory_allocated() / 1024**3:.2f}",
            "reserved_memory_gb": f"{torch.cuda.memory_reserved() / 1024**3:.2f}",
            "free_memory_gb": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f}",
        }
    
    return {
        "status": "running",
        "model_loaded": llm_engine is not None,
        "model_name": getattr(app.state, 'model_name', 'unknown'),
        "gpu": gpu_info
    }


@app.post("/v1/clear_cache")
async def clear_cache():
    """清理 GPU 缓存（显存紧张时调用）"""
    import torch
    import gc
    
    if torch.cuda.is_available():
        # 清理 PyTorch 缓存
        torch.cuda.empty_cache()
        # 强制垃圾回收
        gc.collect()
        
        return {
            "success": True,
            "message": "GPU cache cleared",
            "free_memory_gb": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f}"
        }
    
    return {"success": False, "message": "No GPU available"}


@app.get("/v1/models")
async def list_models():
    """
    列出可用模型（OpenAI 兼容）
    Cherry Studio 等客户端需要此接口
    """
    model_id = app.state.model_name or "figma-json-model"
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "figma-json-service",
                "permission": [],
                "root": model_id,
                "parent": None,
            }
        ]
    }


async def generate_stream(prompt: str, sampling_params: SamplingParams, model_name: str) -> AsyncGenerator[str, None]:
    """真正的 token 级流式生成器"""
    request_id = random_uuid()
    
    # 发送开始事件
    yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    
    # 使用线程池执行同步生成，避免阻塞 event loop
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,  # 使用默认线程池
        lambda: llm_engine.generate(prompt, sampling_params)
    )
    
    if outputs:
        generated_text = outputs[0].outputs[0].text
        parsed_output = parse_output(generated_text)
        
        # 按字符流式发送（逐字输出效果）
        for char in parsed_output:
            chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": char},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            # 小延迟模拟打字效果
            await asyncio.sleep(0.01)
    
    # 发送结束事件
    yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    # 鉴权检查
    await verify_api_key(req)
    """
    OpenAI 兼容的聊天补全接口（支持流式输出）
    
    示例请求:
    ```json
    {
        "model": "figma-json-model",
        "messages": [
            {"role": "user", "content": "创建一个蓝色按钮，宽度100px"}
        ],
        "temperature": 0.1,
        "stream": false
    }
    ```
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 构建 prompt
        prompt = create_chat_prompt(request.messages)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=["<|im_end|>"] + (request.stop or []),
        )
        
        # 流式输出
        if request.stream:
            return StreamingResponse(
                generate_stream(prompt, sampling_params, request.model),
                media_type="text/event-stream"
            )
        
        # 非流式输出（使用线程池避免阻塞 event loop）
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: llm_engine.generate(prompt, sampling_params)
        )
        
        if not outputs:
            raise HTTPException(status_code=500, detail="生成失败")
        
        generated_text = outputs[0].outputs[0].text
        parsed_output = parse_output(generated_text)
        
        # 构建 OpenAI 格式的响应
        response = {
            "id": f"chatcmpl-{random_uuid()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": parsed_output
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids)
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate")
async def generate(request: GenerateRequest, req: Request):
    # 鉴权检查
    await verify_api_key(req)
    """
    Figma JSON 专用生成接口
    
    示例请求:
    ```json
    {
        "instruction": "创建一个蓝色按钮，宽度100px",
        "temperature": 0.1
    }
    ```
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 构建 prompt
        prompt = create_figma_prompt(request.instruction)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=["<|im_end|>"],
        )
        
        # 生成（使用线程池避免阻塞 event loop）
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: llm_engine.generate(prompt, sampling_params)
        )
        
        if not outputs:
            raise HTTPException(status_code=500, detail="生成失败")
        
        generated_text = outputs[0].outputs[0].text
        parsed_output = parse_output(generated_text)
        
        # 尝试解析 JSON
        try:
            json_result = json.loads(parsed_output)
            is_valid_json = True
        except json.JSONDecodeError:
            json_result = None
            is_valid_json = False
        
        return JSONResponse(content={
            "success": True,
            "text": parsed_output,
            "json": json_result,
            "is_valid_json": is_valid_json,
            "usage": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="vLLM 服务部署")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/final_model",
        help="模型路径"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务端口"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="最大序列长度"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="使用 8-bit 量化加载"
    )
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型路径不存在: {args.model_path}")
        print("请先运行训练脚本或指定正确的模型路径")
        sys.exit(1)
    
    # 设置应用状态
    app.state.model_path = args.model_path
    app.state.max_seq_length = args.max_seq_length
    app.state.load_in_8bit = args.load_in_8bit
    
    # 启动服务
    print("\n" + "=" * 60)
    print(f"   启动 vLLM 服务")
    print("=" * 60)
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   模型: {args.model_path}")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
