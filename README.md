# Figma JSON 微调训练框架

基于 Unsloth + Qwen 的 Figma JSON 生成微调项目，覆盖数据处理、LoRA 训练、本地推理和 vLLM 服务部署。

## 当前默认配置

项目当前以代码实现为准，默认配置见 [config.yaml](/home/mario/llm/llmchat/config.yaml)：

- 默认模型：`unsloth/Qwen3-4B`
- 最大序列长度：`2048`
- 量化方式：`8-bit`
- LoRA rank：`32`
- 输出目录：`./outputs`

如果文档和代码不一致，请以 `src/` 和 `config.yaml` 为准。

## 项目结构

```text
.
├── config.yaml
├── requirements.txt
├── start_train.sh
├── run_inference.sh
├── run_vllm_server.sh
├── setup.sh
├── scripts/
│   └── verify_env.py
├── data/
│   └── figma-records.jsonl
└── src/
    ├── data_processor.py
    ├── download_model.py
    ├── inference.py
    ├── train.py
    └── vllm_server.py
```

## 环境要求

- Python 3.10+
- CUDA 11.8+，推荐 CUDA 12.8
- NVIDIA GPU，12GB 显存可跑默认配置

建议先在 `llm` conda 环境里运行。

## 安装依赖

```bash
conda create -n llm python=3.10
conda activate llm

pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 验证环境

```bash
python scripts/verify_env.py
```

## 准备训练数据

`src/data_processor.py` 当前只支持这两个参数：

- `--input`
- `--output`

输入文件支持 `JSON` 数组或 `JSONL`，每条样本至少包含：

```json
{
  "instruction": "创建一个蓝色按钮，宽度 200px，高度 50px",
  "output": "{\"type\":\"RECTANGLE\",\"name\":\"Button\"}"
}
```

处理训练数据：

```bash
python src/data_processor.py --input ./data/figma-records.jsonl --output ./data/train.json
```

如果你有验证集，再单独处理一份：

```bash
python src/data_processor.py --input ./data/raw_val.jsonl --output ./data/val.json
```

`config.yaml` 默认读取：

- `./data/train.json`
- `./data/val.json`

其中验证集是可选的，不存在时会只训练不做验证。

## 开始训练

推荐先激活 conda 环境，再执行：

```bash
./start_train.sh
```

可选参数：

```bash
./start_train.sh --offline
./start_train.sh --mirror
./start_train.sh --help
```

训练入口实际执行的是：

```bash
python src/train.py
```

训练完成后，输出默认在 `outputs/`：

```text
outputs/
├── final_model/
├── merged_16bit/
└── checkpoint-*/
```

## 本地推理

交互模式：

```bash
python src/inference.py --model_path ./outputs/final_model --mode interactive
```

测试样例：

```bash
python src/inference.py --model_path ./outputs/final_model --mode test
```

单次生成：

```bash
python src/inference.py \
  --model_path ./outputs/final_model \
  --mode single \
  --input "创建一个红色圆形，直径 60px" \
  --output result.json
```

## vLLM 服务

启动服务：

```bash
./run_vllm_server.sh ./outputs/final_model 0.0.0.0 8000
```

或直接运行：

```bash
python src/vllm_server.py \
  --model_path ./outputs/final_model \
  --host 0.0.0.0 \
  --port 8000 \
  --max_seq_length 2048
```

服务提供：

- `GET /health`
- `GET /v1/models`
- `GET /v1/status`
- `POST /v1/chat/completions`
- `POST /v1/generate`

## 可下载模型

`src/download_model.py` 当前维护的是 Qwen2.5 备选模型：

```bash
python src/download_model.py --list
python src/download_model.py --model qwen-3b
python src/download_model.py --model qwen-7b
python src/download_model.py --model qwen-14b
```

下载后如果需要切换训练模型，请手动修改 `config.yaml` 的 `model.name`。

## 常见问题

### 1. 训练脚本提示找不到 `python` 或 `torch`

通常是当前 shell 没激活 conda 环境。先确认：

```bash
conda activate llm
which python
python -c "import torch; print(torch.__version__)"
```

### 2. 显存不足

优先调整：

```yaml
model:
  max_seq_length: 1024
  load_in_8bit: true
  load_in_4bit: false

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

如果仍然不够，可以换更小模型，例如 `qwen-3b`。

### 3. HuggingFace 下载慢或失败

可以使用镜像：

```bash
./start_train.sh --mirror
```

或者：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 说明

- 当前文档已按仓库现有代码对齐，不再保留旧版 `--mode sample/process` 之类不存在的命令。
- 如果后续改了 `src/` 或 `config.yaml`，建议同步更新本文档。
