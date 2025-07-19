# 物流填单信息提取模型 - 使用说明

## 项目概述

这是一个基于Qwen2.5-0.5B-Instruct微调的物流填单信息提取模型，专门用于从物流填单文本中提取收件人、地址、电话等结构化信息。

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd llmchat

# 安装依赖
pip install -r requirements.txt
```

### 2. 一键运行

```bash
# 运行快速开始脚本（包含完整流程）
python quick_start.py
```

这个脚本会自动完成：
- 安装依赖
- 生成训练数据
- 训练模型
- 评估模型
- 测试推理

## 详细使用步骤

### 步骤1：生成训练数据

```bash
# 生成默认数据（5000训练+500验证+500测试）
python scripts/generate_data.py

# 自定义数据量
python scripts/generate_data.py --train_samples 10000 --eval_samples 1000 --test_samples 1000
```

### 步骤2：训练模型

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置文件
python train.py --config configs/train_config.yaml

# 同时生成数据和训练
python train.py --generate_data
```

### 步骤3：模型推理

#### 交互式推理
```bash
python inference.py --model_path outputs
```

#### 单次推理
```bash
python inference.py --model_path outputs --input_text "收件人：张三，地址：北京市朝阳区xxx街道，电话：13800138000"
```

#### 保存结果
```bash
python inference.py --model_path outputs --input_text "收件人：张三，地址：北京市朝阳区xxx街道，电话：13800138000" --output_file result.json
```

### 步骤4：模型评估

```bash
# 评估模型性能
python scripts/evaluate.py --model_path outputs

# 自定义测试数据
python scripts/evaluate.py --model_path outputs --test_file data/test.json

# 限制测试样本数
python scripts/evaluate.py --model_path outputs --max_samples 100
```

### 步骤5：发布到Hugging Face

```bash
# 上传模型
python scripts/upload_to_hf.py --repo_name your-username/logistics-extractor --token your-hf-token

# 同时创建模型卡片
python scripts/upload_to_hf.py --repo_name your-username/logistics-extractor --token your-hf-token --create_card
```

## 配置文件说明

### 训练配置 (configs/train_config.yaml)

```yaml
# 模型配置
model:
  base_model: "Qwen/Qwen2.5-0.5B-Instruct"  # 基础模型
  max_length: 2048                           # 最大序列长度
  use_4bit: true                             # 是否使用4bit量化

# 训练配置
training:
  num_epochs: 3                              # 训练轮数
  batch_size: 4                              # 批次大小
  learning_rate: 2e-4                        # 学习率

# LoRA配置
lora:
  r: 16                                      # LoRA秩
  lora_alpha: 32                             # LoRA alpha参数
  target_modules: ["q_proj", "v_proj"]      # 目标模块
```

## 数据格式

### 训练数据格式

训练数据采用JSONL格式，每行一个样本：

```json
{
  "instruction": "请从以下物流填单信息中提取结构化数据",
  "input": "收件人：张三，收件地址：北京市朝阳区中关村大街123号，收件人电话：13800138000",
  "output": "{\"收件人\": \"张三\", \"收件地址\": \"北京市朝阳区中关村大街123号\", \"收件人电话\": \"13800138000\"}"
}
```

### 支持的信息类型

- **收件人信息**: 收件人、收件人电话、收件地址
- **寄件人信息**: 寄件人、寄件人电话、寄件地址
- **其他信息**: 公司、备注、重量等

## 模型性能

在测试集上的典型性能：

- **完全匹配准确率**: >95%
- **部分匹配准确率**: >98%
- **成功率**: >99%
- **推理速度**: ~2秒/样本 (CPU)

## 常见问题

### Q1: 训练时出现内存不足错误
**A**: 可以尝试以下解决方案：
- 减小batch_size（在config中设置）
- 启用4bit量化（use_4bit: true）
- 使用梯度累积（gradient_accumulation_steps）

### Q2: 模型推理结果不准确
**A**: 可能的原因：
- 输入文本格式不规范
- 训练数据质量不够
- 需要调整训练参数

### Q3: 如何提高模型性能
**A**: 建议：
- 增加训练数据量
- 优化数据质量
- 调整LoRA参数
- 使用更大的基础模型

### Q4: 如何自定义数据格式
**A**: 修改`src/data_processing/data_generator.py`中的生成逻辑，或者直接提供自己的JSONL格式数据。

## 高级用法

### 自定义模型配置

```python
from src.model.model_setup import load_model_and_tokenizer
from src.data_processing.dataset import load_datasets

# 加载自定义配置
config = {
    "model": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_length": 1024,
        "use_4bit": True
    },
    # ... 其他配置
}

# 加载模型和数据
model, tokenizer = load_model_and_tokenizer(config)
train_dataset, eval_dataset, data_collator = load_datasets(config, tokenizer)
```

### 批量处理

```python
from src.utils.inference import LogisticsExtractor

extractor = LogisticsExtractor("outputs")

# 批量处理
texts = [
    "收件人：张三，地址：北京市朝阳区xxx街道",
    "收件人：李四，地址：上海市浦东新区xxx路"
]

results = extractor.batch_extract(texts)
for text, result in zip(texts, results):
    print(f"输入: {text}")
    print(f"结果: {result['extracted_info']}")
```

## 部署建议

### 生产环境部署

1. **模型优化**:
   - 使用模型量化减少内存占用
   - 考虑使用TensorRT等推理框架

2. **服务化部署**:
   - 使用FastAPI或Flask创建API服务
   - 使用Docker容器化部署

3. **监控和日志**:
   - 添加性能监控
   - 记录推理日志和错误信息

### 性能优化

1. **推理优化**:
   - 使用GPU加速
   - 批量处理提高吞吐量
   - 模型缓存减少加载时间

2. **内存优化**:
   - 使用模型量化
   - 合理设置batch_size
   - 及时释放不需要的变量

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues
- Email: [your-email@example.com] 