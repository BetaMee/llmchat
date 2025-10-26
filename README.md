# Figma JSON 微调训练框架

基于 Unsloth + Qwen3-8b 的高效微调训练框架，用于训练 Figma JSON 生成模型。

## 项目简介

本项目提供了一个完整的微调训练流程，用于训练一个能够根据自然语言描述生成 Figma JSON 的模型。使用 Unsloth 库加速训练，支持 4-bit 量化和 LoRA 高效微调。

## 特性

- ✅ 基于 Unsloth 的高速训练（比标准训练快 2-5 倍）
- ✅ 支持 Qwen3-8b 系列模型
- ✅ 4-bit 量化降低显存需求
- ✅ LoRA 高效参数微调
- ✅ 完整的数据处理流程
- ✅ 交互式推理模式
- ✅ 多种模型导出格式（LoRA、合并模型、GGUF）

## 环境要求

- Python 3.10+
- CUDA 11.8+ (推荐使用 GPU 训练)
- 16GB+ GPU 显存（使用 4-bit 量化）
- 32GB+ 系统内存

## 快速开始

### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 如果遇到问题，可以分步安装
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets accelerate peft trl bitsandbytes
```

### 2. 准备数据

#### 使用示例数据

```bash
# 生成示例数据并处理
python data_processor.py
```

#### 使用自定义数据

在 `data/` 目录下创建您的数据文件：

**数据格式** (`raw_train.json`):
```json
[
  {
    "input": "创建一个蓝色的矩形按钮，宽度200px，高度50px",
    "output": {
      "type": "RECTANGLE",
      "name": "Button",
      "width": 200,
      "height": 50,
      "fills": [
        {
          "type": "SOLID",
          "color": {"r": 0.0, "g": 0.5, "b": 1.0, "a": 1.0}
        }
      ],
      "cornerRadius": 8
    }
  }
]
```

然后运行数据处理脚本：

```bash
python data_processor.py
```

### 3. 配置训练参数

编辑 `config.yaml` 文件，调整训练参数：

```yaml
# 模型配置
model:
  name: "Qwen/Qwen2.5-8B"
  max_seq_length: 2048
  load_in_4bit: true

# LoRA 配置
lora:
  r: 16
  lora_alpha: 16

# 训练配置
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2.0e-4
```

### 4. 开始训练

```bash
python train.py
```

训练完成后，模型将保存在 `outputs/` 目录：
- `outputs/final_model/` - LoRA 适配器
- `outputs/merged_16bit/` - 合并后的完整模型
- `outputs/gguf_model/` - GGUF 格式（可选）

### 5. 推理测试

#### 交互式模式

```bash
python inference.py --mode interactive
```

#### 测试模式

```bash
python inference.py --mode test
```

#### 单次生成

```bash
python inference.py --mode single --input "创建一个红色圆形，半径50px" --output result.json
```

#### 指定模型路径

```bash
python inference.py --model_path ./outputs/merged_16bit --mode interactive
```

## 项目结构

```
.
├── README.md              # 项目文档
├── requirements.txt       # Python 依赖
├── config.yaml           # 训练配置文件
├── data_processor.py     # 数据处理脚本
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── data/                 # 数据目录
│   ├── raw_train.json    # 原始训练数据
│   ├── raw_val.json      # 原始验证数据
│   ├── train.json        # 处理后的训练数据
│   └── val.json          # 处理后的验证数据
└── outputs/              # 输出目录
    ├── final_model/      # LoRA 模型
    ├── merged_16bit/     # 合并模型
    └── gguf_model/       # GGUF 格式（可选）
```

## 详细说明

### 数据处理

`data_processor.py` 提供了以下功能：

1. **创建示例数据** - 生成用于测试的示例 Figma JSON 数据
2. **数据处理** - 将原始数据转换为训练格式
3. **Prompt 构建** - 使用 Qwen 的对话模板格式化数据

### 训练配置

主要配置参数说明：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `max_seq_length` | 最大序列长度 | 2048 |
| `load_in_4bit` | 使用 4-bit 量化 | true |
| `lora.r` | LoRA rank | 16-64 |
| `learning_rate` | 学习率 | 2e-4 |
| `num_train_epochs` | 训练轮数 | 3-5 |
| `per_device_train_batch_size` | 批次大小 | 2-4 |

### 模型导出

训练完成后会自动导出多种格式：

1. **LoRA 适配器** (`final_model/`) - 仅包含训练的参数，体积小
2. **合并模型** (`merged_16bit/`) - 完整的 16-bit 模型
3. **GGUF 格式** (`gguf_model/`) - 用于 llama.cpp 等工具

### 推理选项

推理脚本支持多种参数：

```python
# 在代码中自定义生成参数
result = inference.generate(
    user_input="创建一个按钮",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

## 性能优化

### 显存优化

1. **使用 4-bit 量化** - 减少显存占用约 75%
2. **调整批次大小** - 根据显存大小调整 `per_device_train_batch_size`
3. **梯度累积** - 使用 `gradient_accumulation_steps` 模拟大批次

### 训练速度

1. **Unsloth 加速** - 自动优化训练速度
2. **混合精度训练** - 使用 bf16 或 fp16
3. **梯度检查点** - 降低显存但会稍微减慢速度

## 常见问题

### Q: 训练时显存不足怎么办？

A: 尝试以下方法：
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 减小 `max_seq_length`
- 确保使用 4-bit 量化

### Q: 如何提高生成质量？

A: 
- 增加训练数据的数量和质量
- 调整 LoRA rank（如 r=32 或 r=64）
- 增加训练轮数
- 调整生成参数（temperature、top_p）

### Q: 支持哪些模型？

A: 支持 Qwen 系列模型，包括：
- Qwen/Qwen2.5-8B
- Qwen/Qwen2.5-14B
- Qwen/Qwen2.5-7B
- 其他兼容的模型

### Q: 如何在 CPU 上运行？

A: 修改 `config.yaml`：
```yaml
model:
  load_in_4bit: false
  dtype: "float32"
```

但 CPU 训练会非常慢，不推荐用于生产环境。

## 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/runs

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
- [Hugging Face](https://huggingface.co/) - 模型和数据集托管

## 更新日志

### v1.0.0 (2024-10-26)
- 初始版本发布
- 支持 Unsloth + Qwen3-8b 微调
- 完整的训练和推理流程
- 示例数据生成
