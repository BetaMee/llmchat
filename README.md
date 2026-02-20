# Figma JSON 微调训练框架

基于 Unsloth + Qwen2.5 的高效微调训练框架，用于训练 Figma JSON 生成模型。

## 项目简介

本项目提供了一个完整的大语言模型微调训练流程，用于训练一个能够根据自然语言描述生成 Figma JSON 的模型。使用 Unsloth 库加速训练（2-5倍提速），支持 4-bit/8-bit 量化和 LoRA 高效微调。

## ✨ 特性

- ✅ **Unsloth 加速训练** - 比标准训练快 2-5 倍
- ✅ **Qwen2.5 系列模型** - 支持 3B/7B/14B 多种规格
- ✅ **量化训练** - 支持 4-bit/8-bit 量化降低显存需求
- ✅ **LoRA 微调** - 高效参数微调（rank=32）
- ✅ **完整数据流程** - 数据处理、训练、推理一体化
- ✅ **灵活部署方式** - 在线/离线/镜像多种模式
- ✅ **环境验证工具** - 自动检测环境配置和网络连接
- ✅ **多格式导出** - LoRA 适配器 + 16-bit 合并模型

## 📋 环境要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **GPU** | NVIDIA RTX 3060 (12GB) | RTX 4070+ (12GB+) |
| **显存** | 12GB（7B 模型 + 4-bit 量化） | 24GB+ |
| **内存** | 16GB | 32GB+ |
| **存储** | 50GB 可用空间 | 100GB+ SSD |

### 软件要求

- **Python**: 3.10+
- **CUDA**: 12.8 (推荐，或 11.8+)
- **PyTorch**: 2.10.0
- **操作系统**: Windows 10/11, Linux, macOS

### 显存与模型对应关系

| 模型 | 量化方式 | 显存需求 | 适用场景 |
|-----|---------|---------|---------|
| Qwen2.5-3B | 4-bit | ~6-8GB | 快速测试、低显存环境 |
| Qwen2.5-7B | 4-bit | ~12-15GB | **推荐**，性价比最高 |
| Qwen2.5-7B | 8-bit | ~8-10GB | 质量与显存平衡 |
| Qwen2.5-14B | 4-bit | ~22-26GB | 高质量需求 |

⚠️ **重要提示**: RTX 4070 (12GB) 训练 7B 模型需要使用 8-bit 量化或降低序列长度到 1024-2048。

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/figma-json-finetuning.git
cd figma-json-finetuning
```

### 2. 安装依赖

```bash
# 创建 conda 环境（推荐）
conda create -n llm python=3.10
conda activate llm

# 安装 PyTorch (CUDA 12.8)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 验证环境

```bash
# 运行环境验证脚本
python scripts/verify_env.py
```

验证脚本会检查：
- Python 版本
- PyTorch 和 CUDA 版本
- GPU 可用性和显存
- Unsloth 和依赖库
- HuggingFace 网络连接

### 4. 准备数据

#### 方式 A：使用示例数据

```bash
# 生成示例数据
python src/data_processor.py --mode sample
```

这将在 `data/` 目录下生成示例训练数据。

#### 方式 B：使用自定义数据

准备你的训练数据文件（支持 JSONL 或 JSON 格式）：

**数据格式** (`data/raw_train.jsonl`):
```jsonl
{"instruction":"创建一个蓝色的矩形按钮，宽度200px，高度50px","input":"","output":"{\"type\":\"RECTANGLE\",\"name\":\"Button\",\"width\":200,\"height\":50,...}"}
{"instruction":"简约扁平风格UI预览界面...","input":"","output":"{\"type\":\"FRAME\",\"name\":\"Preview\",...}"}
```

**字段说明**:
- `instruction`: 设计意图描述（必需）
- `input`: 输入上下文（可留空）
- `output`: 对应的 Figma 节点 JSON（字符串或对象）

然后处理数据：

```bash
# 处理训练数据
python src/data_processor.py --mode process --input ./data/raw_train.jsonl --output ./data/train.json

# 处理验证数据（可选）
python src/data_processor.py --mode process --input ./data/raw_val.jsonl --output ./data/val.json
```

### 5. 配置训练参数

编辑 `config.yaml` 文件：

```yaml
# 模型配置
model:
  name: "unsloth/Qwen2.5-7B-Instruct"  # HuggingFace 模型 ID
  max_seq_length: 4096                  # 最大序列长度
  load_in_4bit: false                   # 是否使用 4-bit 量化
  load_in_8bit: true                    # 是否使用 8-bit 量化（推荐）

# LoRA 配置
lora:
  r: 32                    # LoRA rank（越大越强，但显存占用越多）
  lora_alpha: 32
  lora_dropout: 0

# 训练配置
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
```

**关键配置说明**:

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `max_seq_length` | 最大序列长度，影响显存占用 | 2048-4096 |
| `load_in_8bit` | 8-bit 量化（推荐） | true |
| `load_in_4bit` | 4-bit 量化（更节省显存） | false |
| `lora.r` | LoRA rank，影响训练质量 | 32-64 |
| `gradient_accumulation_steps` | 梯度累积步数，模拟大批次 | 8-16 |

### 6. 开始训练

#### 方式一：在线自动下载（首次推荐）

```bash
./start_train.sh
```

首次运行会自动从 HuggingFace 下载模型到本地缓存（`~/.cache/huggingface/`），之后会复用缓存。

#### 方式二：使用国内镜像加速

```bash
./start_train.sh --mirror
```

使用 HuggingFace 镜像站（hf-mirror.com）加速下载，适合国内网络环境。

#### 方式三：离线模式

```bash
# 先下载模型到本地
python src/download_model.py --model qwen-7b

# 使用离线模式训练
./start_train.sh --offline
```

#### 训练启动脚本参数

```bash
./start_train.sh           # 在线模式（自动下载）
./start_train.sh --offline # 离线模式（使用缓存）
./start_train.sh --mirror  # 使用镜像站
./start_train.sh --help    # 显示帮助
```

### 7. 训练输出

训练完成后，模型会保存在 `outputs/` 目录：

```
outputs/
├── final_model/          # LoRA 适配器（体积小，用于继续训练）
├── merged_16bit/         # 完整的 16-bit 合并模型（用于推理部署）
└── checkpoint-*/         # 训练过程中的检查点
```

### 8. 推理测试

#### 交互式模式

```bash
python src/inference.py --model_path ./outputs/final_model --mode interactive
```

在交互式命令行中输入设计描述，模型会实时生成 Figma JSON。

#### 批量测试模式

```bash
python src/inference.py --model_path ./outputs/final_model --mode test
```

使用预定义的测试样例进行批量测试。

#### 单次生成

```bash
python src/inference.py \
  --model_path ./outputs/final_model \
  --mode single \
  --input "创建一个红色圆形，半径50px" \
  --output result.json
```

## 📁 项目结构

```
.
├── README.md              # 项目文档
├── requirements.txt       # Python 依赖
├── config.yaml           # 训练配置文件
├── .gitignore            # Git 忽略文件
├── start_train.sh        # 训练启动脚本
├── run_inference.sh      # 推理启动脚本
├── src/                  # 源代码目录
│   ├── train.py          # 训练脚本（核心）
│   ├── inference.py      # 推理脚本
│   ├── data_processor.py # 数据处理脚本
│   └── download_model.py # 模型下载工具
├── scripts/              # 工具脚本
│   └── verify_env.py     # 环境验证脚本
├── data/                 # 数据目录
│   ├── train.json        # 处理后的训练数据
│   └── val.json          # 处理后的验证数据
├── outputs/              # 模型输出目录
│   ├── final_model/      # LoRA 适配器
│   ├── merged_16bit/     # 合并模型
│   └── checkpoint-*/     # 训练检查点
└── unsloth_compiled_cache/ # Unsloth 编译缓存（自动生成）
```

## 🔧 详细配置

### 模型下载工具

```bash
# 列出可用模型
python src/download_model.py --list

# 下载 3B 模型
python src/download_model.py --model qwen-3b

# 下载 7B 模型到本地
python src/download_model.py --model qwen-7b --local-dir ./models

# 使用镜像下载（国内）
export HF_ENDPOINT=https://hf-mirror.com
python src/download_model.py --model qwen-7b
```

### 数据处理工具

```bash
# 生成示例数据
python src/data_processor.py --mode sample

# 处理自定义数据
python src/data_processor.py \
  --mode process \
  --input ./data/raw_train.jsonl \
  --output ./data/train.json
```

### 训练配置详解

#### 显存优化配置

如果遇到显存不足（OOM）错误，尝试以下配置：

```yaml
model:
  max_seq_length: 2048      # 从 4096 降低到 2048
  load_in_8bit: true        # 使用 8-bit 量化

data:
  max_length: 2048          # 同步调整数据长度

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # 增加梯度累积
```

#### 训练速度优化

```yaml
training:
  bf16: true                # 使用 bfloat16 混合精度
  gradient_accumulation_steps: 8
  logging_steps: 10

lora:
  use_gradient_checkpointing: unsloth  # 使用 Unsloth 梯度检查点
```

### 支持的模型列表

| 模型 ID | 参数量 | 显存需求 | 特点 |
|---------|--------|---------|------|
| `unsloth/Qwen2.5-3B-Instruct` | 3B | 6-8GB | 轻量快速 |
| `unsloth/Qwen2.5-7B-Instruct` | 7B | 12-15GB | **推荐**，性价比高 |
| `unsloth/Qwen2.5-14B-Instruct` | 14B | 22-26GB | 高质量输出 |

修改 `config.yaml` 中的 `model.name` 即可切换模型。

## 💡 常见问题

### Q1: 训练时显存不足 (CUDA Out of Memory)

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:

1. **降低序列长度**
   ```yaml
   model:
     max_seq_length: 2048  # 从 4096 降到 2048
   data:
     max_length: 2048
   ```

2. **使用更小的模型**
   ```yaml
   model:
     name: unsloth/Qwen2.5-3B-Instruct  # 从 7B 改为 3B
   ```

3. **增加梯度累积**
   ```yaml
   training:
     gradient_accumulation_steps: 16  # 从 8 增加到 16
   ```

4. **使用 8-bit 量化**
   ```yaml
   model:
     load_in_8bit: true
     load_in_4bit: false
   ```

### Q2: Unsloth fused cross entropy 错误

**症状**:
```
RuntimeError: Unsloth: No or negligible GPU memory available for fused cross entropy.
```

**原因**: 7B 模型在 12GB 显存上无法启用 Unsloth 的融合交叉熵优化。

**解决方案**:
- 使用 3B 模型
- 或降低 `max_seq_length` 到 1024-2048
- 或升级到更大显存的 GPU

### Q3: HuggingFace 连接失败

**症状**:
```
ConnectionError: Can't load model from 'unsloth/Qwen2.5-7B-Instruct'
```

**解决方案**:

**方式 1**: 使用镜像站
```bash
./start_train.sh --mirror
```

**方式 2**: 设置环境变量
```bash
export HF_ENDPOINT=https://hf-mirror.com
./start_train.sh
```

**方式 3**: 先下载到本地
```bash
python src/download_model.py --model qwen-7b
./start_train.sh --offline
```

### Q4: 如何提高生成质量？

**建议**:

1. **增加高质量训练数据**
   - 确保数据符合 Figma API 规范
   - 数据多样性（不同类型的 UI 组件）
   - 至少 500+ 条训练样本

2. **调整 LoRA 参数**
   ```yaml
   lora:
     r: 64              # 增加 rank
     lora_alpha: 64
   ```

3. **增加训练轮数**
   ```yaml
   training:
     num_train_epochs: 5  # 从 3 增加到 5
   ```

4. **优化推理参数**
   ```yaml
   inference:
     temperature: 0.1     # 降低随机性
     top_p: 0.9
     max_new_tokens: 2048
   ```

### Q5: 训练过程中断如何恢复？

训练会自动保存检查点到 `outputs/checkpoint-*/`，恢复训练：

```bash
# 修改 train.py 的 TrainingArguments
resume_from_checkpoint="./outputs/checkpoint-100"
```

或者直接从最新检查点继续：

```bash
# 训练脚本会自动检测最新检查点
./start_train.sh
```

### Q6: Windows 上如何运行？

Windows 用户可以使用 Git Bash 或 WSL：

**Git Bash**:
```bash
bash start_train.sh
```

**PowerShell**:
```powershell
# 设置环境变量
$env:HF_ENDPOINT="https://hf-mirror.com"

# 运行训练
python src/train.py
```

**WSL (推荐)**:
```bash
# 在 WSL Ubuntu 中运行
./start_train.sh
```

## 📊 性能优化

### 训练速度优化

| 优化项 | 说明 | 提升 |
|-------|------|------|
| Unsloth 加速 | 自动优化 | 2-5x |
| 混合精度 (bf16) | GPU 计算加速 | 1.5-2x |
| Flash Attention | 注意力机制优化 | 1.5x |
| 梯度检查点 | 显存换时间 | -20% 速度 |

### 显存占用优化

| 优化项 | 显存节省 | 质量影响 |
|-------|---------|---------|
| 4-bit 量化 | 75% | 较小 |
| 8-bit 量化 | 50% | 极小 |
| LoRA | 90% 参数 | 无 |
| 梯度累积 | 50% | 无 |

### 推荐配置组合

**高质量配置** (24GB+ 显存):
```yaml
model:
  name: unsloth/Qwen2.5-14B-Instruct
  max_seq_length: 4096
  load_in_4bit: true

lora:
  r: 64
  
training:
  num_train_epochs: 5
  per_device_train_batch_size: 2
```

**平衡配置** (12GB 显存):
```yaml
model:
  name: unsloth/Qwen2.5-7B-Instruct
  max_seq_length: 2048
  load_in_8bit: true

lora:
  r: 32
  
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

**快速测试配置** (8GB 显存):
```yaml
model:
  name: unsloth/Qwen2.5-3B-Instruct
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 16
  
training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
```

## 📈 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/

# 在浏览器访问
# http://localhost:6006
```

### Weights & Biases

修改 `config.yaml`:
```yaml
training:
  report_to: "wandb"
```

登录并运行训练:
```bash
wandb login
./start_train.sh
```

## 🔗 相关资源

- [Unsloth GitHub](https://github.com/unslothai/unsloth) - 高速 LLM 微调库
- [Qwen 官方](https://github.com/QwenLM/Qwen) - 通义千问大模型
- [Figma API 文档](https://www.figma.com/developers/api) - Figma 开发者文档
- [Hugging Face](https://huggingface.co/) - 模型和数据集托管平台
- [HF Mirror](https://hf-mirror.com/) - HuggingFace 国内镜像

## 📝 更新日志

### v2.2.0 (2025-02-20)

- ✅ 新增 `FigmaJSONTrainer` 类封装训练流程
- ✅ 新增 `check_unsloth_config()` 配置检查方法
- ✅ 新增 `save_models()` 模型保存方法
- ✅ 优化 `verify_env.py` 环境验证工具
- ✅ 修复 RTX 4070 显存不足问题（支持 8-bit 量化）
- ✅ 移除 GGUF 导出功能（Windows 不兼容）
- ✅ 更新项目文档和配置说明

### v2.1.0 (2025-02-20)

- 迁移到 HuggingFace 平台，移除 ModelScope 依赖
- 新增 `start_train.sh` 脚本，支持 `--offline` 和 `--mirror` 参数
- 新增 `verify_env.py` 网络连接检测功能
- 优化项目结构，Python 脚本统一放入 `src/` 目录
- 更新默认模型为 `unsloth/Qwen2.5-7B-Instruct`

### v2.0.0 (2024-12-14)

- 升级到 Qwen2.5 系列模型
- 优化数据处理流程，支持 JSONL 格式
- 改进配置文件和命令行界面
- 增强错误处理和日志输出

### v1.0.0 (2024-10-26)

- 初始版本发布
- 支持 Unsloth + Qwen2.5 微调
- 完整的训练和推理流程
- 示例数据生成

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**: 本项目仅用于学习和研究目的。在生产环境中使用前，请充分测试模型效果和稳定性。
