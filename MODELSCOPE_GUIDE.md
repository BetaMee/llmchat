# ModelScope 使用指南

## 为什么使用 ModelScope？

ModelScope 是阿里云推出的模型社区，提供国内高速下载，无需访问 HuggingFace。

**优势：**
- ✅ 国内高速下载，无墙
- ✅ 稳定可靠的服务
- ✅ 与 HuggingFace 生态兼容
- ✅ 支持所有 Qwen 模型

---

## 快速开始

### 步骤 1: 安装 ModelScope

```bash
pip install modelscope>=1.11.0
```

已包含在 `requirements.txt` 中。

### 步骤 2: 下载模型

#### 方式 A: 使用下载脚本（推荐）

```bash
# 列出所有可用模型
python download_model.py --list

# 下载 Qwen 7B 模型（推荐）
python download_model.py --model qwen-7b

# 下载会自动提示更新 config.yaml
```

#### 方式 B: 交互式下载

```bash
# 直接运行，会显示菜单
python download_model.py
```

#### 方式 C: 手动下载

```python
from modelscope import snapshot_download

# 下载模型
model_dir = snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct',
    cache_dir='./models'
)

print(f"模型路径: {model_dir}")
```

### 步骤 3: 更新配置

下载完成后，更新 `config.yaml`:

```yaml
model:
  name: "./models/Qwen/Qwen2___5-7B-Instruct"  # 你的模型路径
  max_seq_length: 4096
  dtype: null
  load_in_4bit: true
```

或使用脚本自动更新（下载时选择 y）。

### 步骤 4: 开始训练

```bash
./start_train.sh
```

---

## 可用模型

### Qwen 7B（推荐）

```bash
python download_model.py --model qwen-7b
```

- **参数量**: 7B
- **显存需求**: ~12-15GB
- **适合**: A10 (24GB)、A100 (40GB) 等
- **下载大小**: ~4-5GB

### Qwen 14B

```bash
python download_model.py --model qwen-14b
```

- **参数量**: 14B
- **显存需求**: ~22-26GB  
- **适合**: A100 (40GB)、A800 等
- **下载大小**: ~8-10GB

### Qwen 3B（小模型）

```bash
python download_model.py --model qwen-3b
```

- **参数量**: 3B
- **显存需求**: ~6-8GB
- **适合**: 小显存 GPU 或测试
- **下载大小**: ~2-3GB

---

## 完整训练流程

### 1. 准备环境

```bash
# 安装依赖
pip install -r requirements.txt

# 验证环境
python scripts/verify_env.py
```

### 2. 下载模型

```bash
# A10 GPU 推荐 7B 模型
python download_model.py --model qwen-7b
```

下载时选择 `y` 自动更新配置文件。

### 3. 准备数据

```bash
# 处理您的训练数据
python data_processor.py --mode process \
  --input ./data/raw_train.jsonl \
  --output ./data/train.json

# 处理验证数据
python data_processor.py --mode process \
  --input ./data/raw_val.jsonl \
  --output ./data/val.json
```

### 4. 开始训练

```bash
# 使用启动脚本
chmod +x start_train.sh
./start_train.sh

# 或直接运行
python train.py
```

---

## 高级用法

### 指定下载目录

```bash
python download_model.py \
  --model qwen-7b \
  --cache-dir /path/to/your/models
```

### 使用环境变量

```bash
# 设置 ModelScope 缓存目录
export MODELSCOPE_CACHE=./models

# 设置到 ~/.bashrc 永久生效
echo 'export MODELSCOPE_CACHE=./models' >> ~/.bashrc
source ~/.bashrc
```

### 编程方式使用

```python
from modelscope import snapshot_download
import os

# 设置缓存目录
os.environ['MODELSCOPE_CACHE'] = './models'

# 下载模型
model_dir = snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct',
    cache_dir='./models',
    revision='master'
)

# 使用模型
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=4096,
    load_in_4bit=True
)
```

---

## 目录结构

下载后的文件结构：

```
llmchat/
├── models/                          # ModelScope 缓存
│   └── Qwen/
│       └── Qwen2___5-7B-Instruct/  # 下载的模型
│           ├── config.json
│           ├── model.safetensors.index.json
│           ├── model-00001-of-00004.safetensors
│           ├── tokenizer.json
│           └── ...
├── config.yaml                      # 配置文件（已更新）
└── download_model.py                # 下载脚本
```

---

## 常见问题

### Q: 下载速度慢怎么办？

A: ModelScope 已经是国内最快的源，如果还慢：
1. 检查网络连接
2. 尝试不同时间段
3. 使用有线网络而非 WiFi

### Q: 下载中断怎么办？

A: 重新运行下载命令，ModelScope 支持断点续传。

### Q: 下载的模型在哪里？

A: 默认在 `./models/` 目录，可通过以下命令查看：

```bash
ls -lh ./models/Qwen/
```

### Q: 如何删除下载的模型？

A: 直接删除目录：

```bash
rm -rf ./models/Qwen/Qwen2___5-7B-Instruct
```

### Q: 可以同时下载多个模型吗？

A: 可以，但会占用较多磁盘空间。建议先用一个模型测试

。

### Q: 与 HuggingFace 有什么区别？

A: 
- **ModelScope**: 国内快速下载，阿里云支持
- **HuggingFace**: 国际源，需要镜像或代理

模型本身是一样的，都是 Qwen 官方发布。

### Q: 训练时会重新下载吗？

A: 不会。训练时使用本地已下载的模型。

---

## 磁盘空间要求

| 模型 | 下载大小 | 解压后大小 | 推荐空间 |
|------|---------|-----------|---------|
| Qwen 3B | ~2GB | ~4GB | 10GB |
| Qwen 7B | ~4GB | ~8GB | 20GB |
| Qwen 14B | ~8GB | ~16GB | 40GB |

**建议**: 保留 2-3 倍空间用于训练输出和缓存。

---

## 网络问题排查

### 测试 ModelScope 连接

```python
import requests

url = "https://www.modelscope.cn"
try:
    response = requests.get(url, timeout=5)
    print(f"✅ ModelScope 可访问: {response.status_code}")
except Exception as e:
    print(f"❌ ModelScope 不可访问: {e}")
```

### 备选方案

如果 ModelScope 不可用，使用 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 更新模型

当 Qwen 发布新版本时：

```bash
# 删除旧版本
rm -rf ./models/Qwen/Qwen2___5-7B-Instruct

# 重新下载
python download_model.py --model qwen-7b
```

---

## 与 Unsloth 集成

ModelScope 下载的模型可以直接用于 Unsloth：

```python
from unsloth import FastLanguageModel

# ModelScope 本地模型路径
model_path = "./models/Qwen/Qwen2___5-7B-Instruct"

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,  # 本地路径
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True
)
```

---

## 支持和反馈

- **ModelScope 文档**: https://modelscope.cn/docs
- **Qwen 模型**: https://modelscope.cn/organization/Qwen
- **项目 Issues**: 遇到问题请在项目中提 Issue

---

## 总结

使用 ModelScope 的完整流程：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型
python download_model.py --model qwen-7b

# 3. 处理数据
python data_processor.py --mode process \
  --input ./data/raw_train.jsonl \
  --output ./data/train.json

# 4. 开始训练
./start_train.sh
```

**简单、快速、无需翻墙！** 🎉
