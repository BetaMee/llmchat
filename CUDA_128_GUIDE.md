# CUDA 12.8 兼容性指南

## ⚠️ 重要问题

**CUDA 12.8 + bitsandbytes 兼容性问题**

您遇到的错误：
```
RuntimeError: CUDA Setup failed despite GPU being available
Required library version not found: libbitsandbytes_cuda128.so
```

**原因**：bitsandbytes 目前没有为 CUDA 12.8 预编译的库文件。

---

## ✅ 解决方案

### 方案 1: 使用 BF16 训练（推荐，已配置）

**优势**：
- ✅ 无需 4-bit 量化
- ✅ 开箱即用，无需编译
- ✅ 训练速度快
- ✅ A10 24GB 显存足够

**配置已更新**：
```yaml
# config.yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  dtype: "bfloat16"  # 使用 bf16
  load_in_4bit: false  # 禁用量化
```

**显存占用**（7B 模型 BF16）：
- 模型：~14GB
- 训练：~8-10GB
- **总计**：~22-24GB ✅ A10 刚好够用

**立即使用**：
```bash
# 配置已自动更新，直接下载模型
python download_model.py --model qwen-7b

# 开始训练
python train.py
```

---

### 方案 2: 使用更小的模型

如果显存紧张，使用 3B 模型：

```bash
# 下载 3B 模型
python download_model.py --model qwen-3b

# 更新 config.yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
```

**显存占用**（3B 模型 BF16）：
- 模型：~6GB
- 训练：~4-6GB
- **总计**：~10-12GB ✅ 余量充足

---

### 方案 3: 从源码编译 bitsandbytes（高级）

**仅在需要 4-bit 量化时使用**

```bash
# 1. 克隆仓库
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes

# 2. 编译（CUDA 12.8）
CUDA_VERSION=128 python setup.py install

# 3. 验证
python -m bitsandbytes
```

**注意**：
- ⏰ 编译需要 15-30 分钟
- 需要 CUDA 开发工具
- 可能遇到其他兼容性问题

**不推荐** 此方案，因为 BF16 已足够好。

---

## 📊 训练方案对比

| 方案 | 显存 | 速度 | 难度 | 推荐度 |
|------|------|------|------|--------|
| **BF16 (7B)** | 22-24GB | 快 | ⭐ 简单 | ⭐⭐⭐⭐⭐ |
| **BF16 (3B)** | 10-12GB | 很快 | ⭐ 简单 | ⭐⭐⭐⭐ |
| **4-bit (编译)** | 12-15GB | 中等 | ⭐⭐⭐⭐⭐ 复杂 | ⭐⭐ |

---

## 🚀 推荐流程（BF16 训练）

### 步骤 1: 下载模型

```bash
# 使用 ModelScope 下载 7B 模型
python download_model.py --model qwen-7b

# 或交互式选择
python download_model.py
```

### 步骤 2: 准备数据

```bash
# 处理训练数据
python data_processor.py --mode process \
  --input ./data/raw_train.jsonl \
  --output ./data/train.json

# 处理验证数据  
python data_processor.py --mode process \
  --input ./data/raw_val.jsonl \
  --output ./data/val.json
```

### 步骤 3: 开始训练

```bash
# 配置已优化，直接运行
python train.py
```

---

## 🔍 显存监控

训练时监控 GPU 显存：

```bash
# 方式 1: 使用 nvidia-smi
watch -n 1 nvidia-smi

# 方式 2: 在 Python 中查看
python -c "
import torch
print(f'GPU 总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'已用显存: {torch.cuda.memory_allocated() / 1e9:.1f} GB')
print(f'保留显存: {torch.cuda.memory_reserved() / 1e9:.1f} GB')
"
```

---

## ⚙️ 优化配置

### 如果显存不足

**选项 1: 降低 batch size**
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # 增加梯度累积
```

**选项 2: 缩短序列长度**
```yaml
model:
  max_seq_length: 2048  # 从 4096 降到 2048
```

**选项 3: 使用梯度检查点**
```yaml
lora:
  use_gradient_checkpointing: "unsloth"  # 已启用
```

### 如果显存充足

**可以增加 batch size**：
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
```

---

## 🎯 BF16 vs 4-bit 对比

| 特性 | BF16 | 4-bit |
|------|------|-------|
| 显存占用 | 2x 模型大小 | 0.25x 模型大小 |
| 训练速度 | 快 | 中等 |
| 精度 | 高 | 稍低 |
| 兼容性 | ✅ 好 | ⚠️ 依赖 bitsandbytes |
| CUDA 12.8 | ✅ 支持 | ❌ 不支持 |

**结论**：在 CUDA 12.8 环境下，BF16 是最佳选择。

---

## ✅ 验证配置

运行以下命令验证设置：

```python
python -c "
import torch
import yaml

# 检查 GPU
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'✅ CUDA: {torch.version.cuda}')
print(f'✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# 检查 BF16 支持
bf16_support = torch.cuda.is_bf16_supported()
print(f'✅ BF16 支持: {'是' if bf16_support else '否'}')

# 检查配置
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    print(f'✅ 量化模式: {config['model']['load_in_4bit']}')
    print(f'✅ 数据类型: {config['model']['dtype']}')

print('\n配置正确，可以开始训练！')
"
```

---

## 📝 训练启动检查清单

- [x] 配置已更新（load_in_4bit: false）
- [ ] 模型已下载（python download_model.py）
- [ ] 数据已准备（train.json + val.json）
- [ ] GPU 显存充足（至少 22GB 可用）

---

## 🆘 故障排除

### 问题 1: 显存不足（OOM）

```bash
# 解决方案：降低 batch size
# 修改 config.yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### 问题 2: 模型下载失败

```bash
# 使用 ModelScope（国内快）
python download_model.py --model qwen-7b

# 或手动设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 3: 训练速度慢

```bash
# 确认使用 BF16
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    print(f\"BF16: {config['training']['bf16']}\")
"

# 应该输出：BF16: True
```

---

## 🎉 准备就绪

现在您可以：

```bash
# 1. 下载模型（自动使用 BF16 配置）
python download_model.py --model qwen-7b

# 2. 准备数据
python data_processor.py --mode process \
  --input ./data/your_train.jsonl \
  --output ./data/train.json

# 3. 开始训练
python train.py
```

**无需担心 bitsandbytes 问题，BF16 训练完全正常！** ✅

---

## 📚 相关文档

- [ModelScope 使用指南](MODELSCOPE_GUIDE.md)
- [网络配置指南](NETWORK_SETUP.md)
- [主项目 README](README.md)

---

## 💡 总结

**CUDA 12.8 最佳实践**：
1. 使用 BF16 训练（已配置）
2. 禁用 4-bit 量化
3. 下载完整模型（非bnb版本）
4. 确保 22-24GB 显存可用

**开箱即用，无需编译！** 🚀
