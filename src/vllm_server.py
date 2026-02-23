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
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams
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
    
    llm_engine = LLM(
        model=model_path,
        max_model_len=max_seq_length,
        dtype="bfloat16",
        quantization="bitsandbytes" if app.state.load_in_8bit else None,
        load_format="auto",
        trust_remote_code=True,
    )
    
    print("✅ 模型加载完成!")
    print(f"   模型路径: {model_path}")
    print(f"   最大序列长度: {max_seq_length}")
    print("=" * 60)
    
    yield
    
    # 关闭时清理
    print("\n正在关闭服务...")


app = FastAPI(
    title="Figma JSON vLLM API",
    description="基于 vLLM 的 Figma JSON 生成服务",
    version="1.0.0",
    lifespan=lifespan
)


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
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "model_loaded": llm_engine is not None}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天补全接口
    
    示例请求:
    ```json
    {
        "model": "figma-json-model",
        "messages": [
            {"role": "user", "content": "创建一个蓝色按钮，宽度100px"}
        ],
        "temperature": 0.1
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
        
        # 生成
        outputs = llm_engine.generate(prompt, sampling_params)
        
        if not outputs:
            raise HTTPException(status_code=500, detail="生成失败")
        
        generated_text = outputs[0].outputs[0].text
        parsed_output = parse_output(generated_text)
        
        # 构建 OpenAI 格式的响应
        response = {
            "id": f"chatcmpl-{random_uuid()}",
            "object": "chat.completion",
            "created": int(__import__('time').time()),
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
async def generate(request: GenerateRequest):
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
        
        # 生成
        outputs = llm_engine.generate(prompt, sampling_params)
        
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
