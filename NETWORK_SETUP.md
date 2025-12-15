# 网络配置和问题解决

## 问题诊断

您遇到的错误：
```
Network is unreachable - Failed to connect to huggingface.co
```

这是网络访问问题。以下是解决方案：

---

## 解决方案 1: 配置 HuggingFace 镜像（推荐）

### 方法 A: 使用魔搭社区镜像

```bash
# 设置环境变量（临时）
export HF_ENDPOINT=https://hf-mirror.com

# 或者设置到 ~/.bashrc 或 ~/.zshrc（永久）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 方法 B: 使用 ModelScope

安装 ModelScope：
```bash
pip install modelscope
```

修改 `train.py` 使用 ModelScope：

```python
# 在文件顶部添加
import os
os.environ['USE_MODELSCOPE'] = '1'

# 或在加载模型前设置
from modelscope import snapshot_download

# 下载模型
model_path = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./models')
```

---

## 解决方案 2: 使用本地模型

### 步骤 1: 下载模型到本地

如果您已经有模型文件，将其放在项目目录：

```
llmchat/
└── models/
    └── Qwen2.5-7B-Instruct/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

### 步骤 2: 修改配置

```yaml
# config.yaml
model:
  name: "./models/Qwen2.5-7B-Instruct"  # 本地路径
  max_seq_length: 4096
  dtype: null
  load_in_4bit: true
```

---

## 解决方案 3: 修复 Import 顺序警告

修改 `train.py` 的导入顺序：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsloth + Qwen 微调训练脚本
"""

import os
import json
import yaml
import torch

# ⚠️ 重要：Unsloth 必须在其他库之前导入
from unsloth import FastLanguageModel

# 然后导入其他库
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# ... 其余代码
```

---

## 完整的训练启动脚本

创建 `start_train.sh`：

```bash
#!/bin/bash

# 配置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 可选：配置代理（如果您有）
# export HTTP_PROXY=http://your-proxy:port
# export HTTPS_PROXY=http://your-proxy:port

# 可选：离线模式（使用本地模型）
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# 启动训练
python train.py
```

使用方式：
```bash
chmod +x start_train.sh
./start_train.sh
```

---

## 推荐的模型选择

根据您的 GPU 显存选择：

### A10 GPU (24GB 显存)

**推荐模型：**
```yaml
# config.yaml
model:
  name: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  # 或本地路径
  # name: "./models/Qwen2.5-7B-Instruct"
```

**显存占用估算：**
- 模型：~4-5GB (4-bit 量化)
- 训练：~8-10GB
- 总计：~12-15GB ✅ 足够

### 如果有更多显存

**14B 模型：**
```yaml
model:
  name: "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
```

**显存占用估算：**
- 模型：~7-8GB
- 训练：~16-20GB
- 总计：~23-28GB ⚠️ 接近极限

---

## 使用 ModelScope 下载模型

创建 `download_model.py`：

```python
#!/usr/bin/env python3
"""从 ModelScope 下载模型"""

from modelscope import snapshot_download

# 下载 Qwen2.5-7B-Instruct
model_dir = snapshot_download(
    'qwen/Qwen2.5-7B-Instruct',
    cache_dir='./models',
    revision='master'
)

print(f"模型已下载到: {model_dir}")
print(f"\n更新 config.yaml:")
print(f'model:\n  name: "{model_dir}"')
```

运行：
```bash
python download_model.py
```

---

## 验证网络配置

创建 `test_network.py`：

```python
#!/usr/bin/env python3
"""测试网络连接"""

import requests
import os

def test_connection(url, name):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: 连接成功")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

print("测试网络连接...\n")

# 测试各种源
sources = [
    ("https://huggingface.co", "HuggingFace 官方"),
    ("https://hf-mirror.com", "HuggingFace 镜像"),
    ("https://www.modelscope.cn", "ModelScope"),
]

for url, name in sources:
    test_connection(url, name)

# 显示当前环境变量
print(f"\n当前环境变量:")
print(f"HF_ENDPOINT: {os.getenv('HF_ENDPOINT', '未设置')}")
print(f"HTTP_PROXY: {os.getenv('HTTP_PROXY', '未设置')}")
```

运行：
```bash
python test_network.py
```

---

## 快速修复步骤

### 最简单的方法：

1. **设置镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

2. **使用更小的模型**
```yaml
# config.yaml
model:
  name: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
```

3. **重新运行训练**
```bash
python train.py
```

---

## 常见问题

### Q: 还是连接不上怎么办？

A: 使用完全离线模式：

1. 在其他有网络的机器下载模型
2. 传输到训练机器
3. 使用本地路径：
```yaml
model:
  name: "./models/Qwen2.5-7B-Instruct"
```

### Q: ModelScope 下载很慢？

A: 使用镜像加速：
```python
os.environ['MODELSCOPE_CACHE'] = './models'
```

### Q: Import 顺序警告影响训练吗？

A: 会影响性能。必须先导入 unsloth。

---

## 联系方式

如果以上方法都不行，请：
1. 检查防火墙设置
2. 确认网络代理配置
3. 尝试使用 VPN/代理
