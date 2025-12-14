# macOS 安装指南

## ⚠️ macOS 兼容性说明

这个项目在 macOS 上安装时会遇到一些依赖兼容性问题，主要是 `bitsandbytes` 库的限制。

## 🔍 问题说明

1. **bitsandbytes 版本限制**
   - macOS 上的 bitsandbytes 最高版本是 0.42.0
   - 项目原本要求 bitsandbytes>=0.43.0
   - unsloth[colab-new] 要求 bitsandbytes>=0.45.5

2. **依赖冲突**
   ```
   unsloth[colab-new] 依赖 bitsandbytes>=0.45.5
   但 macOS 只支持 bitsandbytes<=0.42.0
   ```

## ✅ 解决方案

### 方案 1: 使用基础版 unsloth（推荐用于 macOS）

已经修改 `requirements.txt` 为 macOS 兼容版本：

```txt
# 使用基础版 unsloth（不带 [colab-new]）
unsloth @ git+https://github.com/unslothai/unsloth.git

# 限制 bitsandbytes 版本
bitsandbytes>=0.42.0,<0.43.0
```

**安装命令**：
```bash
pip install -r requirements.txt
```

**注意**: 
- ⚠️ macOS 上的训练速度可能比 Linux/Windows 慢
- ⚠️ 某些高级优化功能可能不可用
- ⚠️ 建议在有 GPU 的 Linux 机器上进行实际训练

### 方案 2: 分步安装（如果方案1失败）

```bash
# 1. 先安装核心依赖
pip install torch transformers datasets accelerate peft trl

# 2. 安装 bitsandbytes（macOS 版本）
pip install bitsandbytes==0.42.0

# 3. 安装基础版 unsloth
pip install git+https://github.com/unslothai/unsloth.git

# 4. 安装其他工具
pip install wandb tensorboard scikit-learn pandas numpy tqdm ipython jupyter
```

### 方案 3: 仅安装推理和评估所需的依赖

如果你只需要运行推理或评估（不需要训练）：

```bash
# 最小依赖安装
pip install torch transformers datasets accelerate
pip install openai  # 用于评估脚本
```

### 方案 4: 使用 Docker（推荐用于生产环境）

创建 `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip git

WORKDIR /app
COPY requirements.txt .

# 使用 Linux 版本的 requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

运行：
```bash
docker build -t figma-json-trainer .
docker run --gpus all -v $(pwd)/data:/app/data figma-json-trainer
```

### 方案 5: 使用云平台（最推荐）

在有 GPU 的云平台上运行：

#### Google Colab
```python
# 在 Colab notebook 中
!git clone https://github.com/yourusername/llmchat.git
%cd llmchat
!pip install -r requirements.txt
!python train.py
```

#### Kaggle Notebooks
上传代码和数据到 Kaggle，使用免费的 GPU

#### AWS/Azure/GCP
租用 GPU 实例，在 Linux 环境下运行

## 📊 不同环境对比

| 环境 | 训练速度 | 显存需求 | 兼容性 | 推荐度 |
|------|---------|---------|--------|--------|
| **macOS (Apple Silicon)** | 慢 | 高 | ⚠️ 有限制 | ⭐⭐ (仅开发测试) |
| **macOS (Intel)** | 很慢 | 高 | ⚠️ 有限制 | ⭐ (不推荐) |
| **Linux + NVIDIA GPU** | 快 | 低 (量化后) | ✅ 完全 | ⭐⭐⭐⭐⭐ |
| **Windows + NVIDIA GPU** | 快 | 低 (量化后) | ✅ 完全 | ⭐⭐⭐⭐ |
| **Google Colab** | 快 | 低 | ✅ 完全 | ⭐⭐⭐⭐⭐ |
| **Kaggle** | 快 | 低 | ✅ 完全 | ⭐⭐⭐⭐ |

## 🎯 针对 macOS 用户的建议

### 开发阶段（本地 macOS）
```bash
# 仅安装推理和评估依赖
pip install torch transformers datasets accelerate openai

# 使用以下脚本进行开发测试
python data_processor.py  # 数据处理
python inference.py --mode test  # 推理测试
python eval/eval_figma.py --create_sample  # 创建测试数据
```

### 训练阶段（云平台）
使用 Google Colab 或其他云 GPU 服务

### 评估阶段（本地或云端）
```bash
# 部署模型到云端，本地运行评估
export API_BASE_URL="https://your-api-endpoint"
python eval/eval_figma.py
```

## 🛠️ macOS 特定问题解决

### 问题 1: Torch 安装失败
```bash
# 使用 CPU 版本
pip install torch torchvision torchaudio
```

### 问题 2: 找不到 CUDA
```bash
# macOS 不支持 CUDA，使用 MPS (Metal Performance Shaders)
# 或 CPU 模式，修改 config.yaml:
model:
  load_in_4bit: false
  dtype: "float32"
```

### 问题 3: 内存不足
```bash
# 减小批次大小
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

## 📝 验证安装

运行以下命令验证安装：

```bash
# 测试 Python 导入
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 测试数据处理
python data_processor.py

# 检查可用设备
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

## 💡 推荐工作流程

### macOS 上的开发流程

1. **本地开发**
   ```bash
   # 数据准备和处理
   python data_processor.py
   
   # 代码调试
   python inference.py --mode test
   ```

2. **云端训练**
   ```bash
   # 上传到 Google Colab
   # 运行训练脚本
   # 下载训练好的模型
   ```

3. **本地评估**
   ```bash
   # 部署模型或使用 API
   python eval/eval_figma.py
   ```

## 🔗 相关资源

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [bitsandbytes 兼容性](https://github.com/TimDettmers/bitsandbytes/issues)
- [Google Colab](https://colab.research.google.com/)
- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)

## 📞 获取帮助

如果遇到问题：
1. 检查是否使用了正确的 requirements.txt（macOS 版本）
2. 尝试分步安装方案
3. 考虑使用云平台进行训练
4. 仅在本地进行开发和测试，训练放在云端

## ⚡ 快速开始（macOS 用户）

```bash
# 1. 克隆项目
git clone <your-repo>
cd llmchat

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 最小化安装（开发用）
pip install torch transformers datasets accelerate openai

# 4. 准备数据
python data_processor.py

# 5. 测试推理（使用预训练模型）
python inference.py --mode test

# 6. 在云端训练模型
# (使用 Google Colab 或其他云服务)

# 7. 评估模型
python eval/eval_figma.py --create_sample
```

---

**总结**: macOS 可以用于开发和测试，但强烈建议在有 NVIDIA GPU 的 Linux 系统或云平台上进行实际训练。
