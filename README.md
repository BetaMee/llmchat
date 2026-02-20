# Figma JSON 微调训练框架

基于 Unsloth + Qwen2.5 的高效微调训练框架，用于训练 Figma JSON 生成模型。

## 项目简介

本项目提供了一个完整的微调训练流程，用于训练一个能够根据自然语言描述生成 Figma JSON 的模型。使用 Unsloth 库加速训练，支持 4-bit 量化和 LoRA 高效微调。

## 特性

- ✅ 基于 Unsloth 的高速训练（比标准训练快 2-5 倍）
- ✅ 支持 Qwen2.5 系列模型（7B/14B 推荐）
- ✅ 4-bit 量化降低显存需求
- ✅ LoRA 高效参数微调（rank=32）
- ✅ 完整的数据处理流程
- ✅ 交互式推理模式
- ✅ 多种模型导出格式（LoRA、合并模型、GGUF）
- ✅ 支持 HuggingFace 在线/离线模式
- ✅ 国内镜像支持（hf-mirror.com）

## 环境要求

- Python 3.10+
- CUDA 11.8+ (推荐使用 GPU 训练)
- 12GB+ GPU 显存（使用 4-bit 量化训练 7B 模型）
- 24GB+ GPU 显存（训练 14B 模型）
- 32GB+ 系统内存

## 快速开始

### 1. 环境验证

```bash
# 验证环境配置
python scripts/verify_env.py
```

### 2. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt
```

### 3. 准备数据

#### 使用示例数据

```bash
# 生成示例数据并处理
python src/data_processor.py --mode sample
```

#### 使用自定义数据

准备你的训练数据文件（支持 JSONL 或 JSON 格式）：

**数据格式** (`data/raw_train.jsonl`):
```jsonl
{"instruction":"创建一个蓝色的矩形按钮，宽度200px，高度50px","input":"","output":"{\"type\":\"RECTANGLE\",\"name\":\"Button\",\"width\":200,\"height\":50,...}"}
{"instruction":"简约扁平风格UI预览界面...","input":"","output":"{\"type\":\"FRAME\",\"name\":\"Preview\",...}"}
```

**字段说明**:
- `instruction`: 设计意图描述（必需）
- `input`: 输入上下文（当前版本可留空）
- `output`: 对应的 Figma 节点 JSON（可以是 JSON 字符串或对象）

然后运行数据处理脚本：

```bash
# 处理训练数据
python src/data_processor.py --mode process --input ./data/raw_train.jsonl --output ./data/train.json

# 处理验证数据
python src/data_processor.py --mode process --input ./data/raw_val.jsonl --output ./data/val.json
```

### 4. 配置训练参数

编辑 `config.yaml` 文件，调整训练参数：

```yaml
# 模型配置
model:
  name: "unsloth/Qwen2.5-7B-Instruct"  # HuggingFace 模型 ID
  max_seq_length: 4096
  load_in_4bit: true

# LoRA 配置
lora:
  r: 32  # LoRA rank
  lora_alpha: 32

# 训练配置
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
```

### 5. 开始训练

#### 方式一：在线自动下载（推荐首次使用）

```bash
# 自动从 HuggingFace 下载模型并训练
./start_train.sh
```

#### 方式二：国内镜像加速

```bash
# 使用 HuggingFace 镜像站
./start_train.sh --mirror
```

#### 方式三：先下载到本地，再离线训练

```bash
# 下载模型到 ./models/ 目录
python src/download_model.py --model qwen-7b

# 离线模式运行训练
./start_train.sh --offline
```

训练完成后，模型将保存在 `outputs/` 目录：
- `outputs/final_model/` - LoRA 适配器
- `outputs/merged_16bit/` - 合并后的完整模型
- `outputs/gguf_model/` - GGUF 格式（可选）

### 6. 推理测试

#### 交互式模式

```bash
python src/inference.py --model_path ./outputs/final_model --mode interactive
```

#### 测试模式

```bash
python src/inference.py --model_path ./outputs/final_model --mode test
```

#### 单次生成

```bash
python src/inference.py \
  --model_path ./outputs/final_model \
  --mode single \
  --input "创建一个红色圆形，半径50px" \
  --output result.json
```

## 项目结构

```
.
├── README.md              # 项目文档
├── requirements.txt       # Python 依赖
├── config.yaml           # 训练配置文件
├── start_train.sh        # 训练启动脚本（支持 --offline/--mirror 参数）
├── run_inference.sh      # 推理启动脚本
├── src/                  # 源代码目录
│   ├── data_processor.py # 数据处理脚本
│   ├── train.py          # 训练脚本
│   ├── inference.py      # 推理脚本
│   └── download_model.py # 模型下载脚本
├── scripts/              # 工具脚本
│   └── verify_env.py     # 环境验证脚本
├── data/                 # 数据目录
│   ├── raw_train.jsonl   # 原始训练数据
│   ├── raw_val.jsonl     # 原始验证数据
│   ├── train.json        # 处理后的训练数据
│   └── val.json          # 处理后的验证数据
├── models/               # 本地模型目录（可选）
└── outputs/              # 输出目录
    ├── final_model/      # LoRA 模型
    ├── merged_16bit/     # 合并模型
    └── gguf_model/       # GGUF 格式（可选）
```

## 详细说明

### 训练启动脚本参数

`start_train.sh` 支持以下参数：

```bash
# 在线模式（自动下载模型）
./start_train.sh

# 离线模式（使用本地缓存）
./start_train.sh --offline

# 使用 HuggingFace 镜像（国内访问）
./start_train.sh --mirror

# 离线 + 镜像（组合使用）
./start_train.sh --offline --mirror

# 显示帮助
./start_train.sh --help
```

### 模型下载

```bash
# 列出可用模型
python src/download_model.py --list

# 下载 7B 模型到本地
python src/download_model.py --model qwen-7b

# 下载到指定目录
python src/download_model.py --model qwen-7b --local-dir /path/to/models

# 使用镜像下载（国内）
export HF_ENDPOINT=https://hf-mirror.com
python src/download_model.py --model qwen-7b
```

### 数据处理

`data_processor.py` 提供了以下功能：

1. **创建示例数据** - 生成用于测试的示例 Figma JSON 数据
2. **数据处理** - 将原始数据转换为训练格式
3. **Prompt 构建** - 使用 Qwen 的对话模板格式化数据
4. **灵活格式** - 支持 JSONL 和 JSON 两种输入格式

**命令行参数**：
```bash
python src/data_processor.py --mode [sample|process] --input <输入文件> --output <输出文件>
```

### 训练配置

主要配置参数说明：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `model.name` | HuggingFace 模型 ID | `unsloth/Qwen2.5-7B-Instruct` |
| `max_seq_length` | 最大序列长度 | 4096 |
| `load_in_4bit` | 使用 4-bit 量化 | true |
| `lora.r` | LoRA rank | 32 |
| `learning_rate` | 学习率 | 1e-4 |
| `num_train_epochs` | 训练轮数 | 3-5 |
| `per_device_train_batch_size` | 批次大小 | 1 |
| `gradient_accumulation_steps` | 梯度累积步数 | 8 |

### 支持的模型

当前推荐使用的模型：

| 模型 | 显存需求 | 适用场景 |
|------|---------|---------|
| `unsloth/Qwen2.5-3B-Instruct` | ~6-8GB | 快速测试、低显存 |
| `unsloth/Qwen2.5-7B-Instruct` | ~12-15GB | **推荐**，性价比最高 |
| `unsloth/Qwen2.5-14B-Instruct` | ~22-26GB | 高质量需求 |

修改 `config.yaml` 中的 `model.name` 即可切换模型。

### 模型导出

训练完成后会自动导出多种格式：

1. **LoRA 适配器** (`final_model/`) - 仅包含训练的参数，体积小
2. **合并模型** (`merged_16bit/`) - 完整的 16-bit 模型
3. **GGUF 格式** (`gguf_model/`) - 用于 llama.cpp 等工具（可选）

## 性能优化

### 显存优化

1. **使用 4-bit 量化** - 减少显存占用约 75%
2. **调整批次大小** - 根据显存大小调整 `per_device_train_batch_size`
3. **梯度累积** - 使用 `gradient_accumulation_steps` 模拟大批次
4. **梯度检查点** - 使用 Unsloth 的梯度检查点功能

### 训练速度

1. **Unsloth 加速** - 自动优化训练速度（2-5倍提升）
2. **混合精度训练** - 使用 bf16 混合精度
3. **合理的序列长度** - 根据数据复杂度调整 `max_seq_length`

## 常见问题

### Q: 训练时显存不足怎么办？

A: 尝试以下方法：
- 使用更小的模型（7B → 3B）
- 减小 `per_device_train_batch_size` 为 1
- 增加 `gradient_accumulation_steps` 到 16
- 减小 `max_seq_length` 到 2048
- 确保使用 4-bit 量化

### Q: HuggingFace 连接失败怎么办？

A: 国内用户可以使用镜像：
```bash
# 方式一：使用脚本参数
./start_train.sh --mirror

# 方式二：设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
./start_train.sh
```

### Q: 如何提高生成质量？

A: 
- 增加训练数据的数量和质量
- 调整 LoRA rank（如 r=64）
- 增加训练轮数（num_train_epochs=5）
- 调整生成参数（temperature 降低到 0.1）
- 确保训练数据符合 Figma API 规范

### Q: 如何在 CPU 上运行？

A: 修改 `config.yaml`：
```yaml
model:
  load_in_4bit: false
  dtype: "float32"
```

但 CPU 训练会非常慢，不推荐用于生产环境。

## 数据格式示例

### 完整的训练样本

```json
{
  "instruction": "简约扁平风格UI预览界面，三列等宽圆角矩形卡片，浅灰填充，深灰背景，左上角\"Preview\"文字标签",
  "input": "",
  "output": {
    "type": "FRAME",
    "name": "Preview",
    "width": 327,
    "height": 93,
    "x": 24,
    "y": 589,
    "blendMode": "PASS_THROUGH",
    "children": [
      {
        "type": "TEXT",
        "name": "Preview",
        "characters": "Preview",
        "fontSize": 14,
        "fontName": {
          "family": "Inter",
          "style": "Medium"
        }
      }
    ]
  }
}
```

## 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/

# 在浏览器访问
# http://localhost:6006
```

### Weights & Biases

修改 `config.yaml`：
```yaml
training:
  report_to: "wandb"
```

然后在训练前登录：
```bash
wandb login
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 相关资源

- [Unsloth](https://github.com/unslothai/unsloth) - 高速 LLM 微调库
- [Qwen](https://github.com/QwenLM/Qwen) - 通义千问大模型
- [Figma API](https://www.figma.com/developers/api) - Figma 开发者文档
- [Hugging Face](https://huggingface.co/) - 模型和数据集托管

## 更新日志

### v2.1.0 (2025-02-20)
- 迁移到 HuggingFace 平台，移除 ModelScope 依赖
- 新增 `start_train.sh` 脚本，支持 `--offline` 和 `--mirror` 参数
- 新增 `verify_env.py` 网络连接检测功能
- 优化项目结构，Python 脚本统一放入 `src/` 目录
- 更新默认模型为 `unsloth/Qwen2.5-7B-Instruct`

### v2.0.0 (2024-12-14)
- 升级到 Qwen3-VL-32B-Instruct 模型
- 优化数据处理流程，支持 JSONL 格式
- 移除评估脚本，专注于训练和推理
- 改进配置文件和命令行界面
- 增强错误处理和日志输出

### v1.0.0 (2024-10-26)
- 初始版本发布
- 支持 Unsloth + Qwen2.5-8B 微调
- 完整的训练和推理流程
- 示例数据生成
