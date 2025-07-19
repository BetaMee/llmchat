---
language:
- zh
license: mit
tags:
- logistics
- information-extraction
- qwen
- chinese
- structured-data
datasets:
- custom-logistics-dataset
metrics:
- accuracy
- f1-score
pipeline_tag: text-generation
---

# 物流填单信息提取模型 (Logistics Information Extractor)

基于Qwen2.5-0.5B-Instruct微调的物流填单信息结构化提取模型，专门用于从物流填单文本中提取收件人、地址、电话等关键信息。

## 模型描述

- **开发者**: [您的名字]
- **基础模型**: Qwen2.5-0.5B-Instruct
- **微调方法**: LoRA (Low-Rank Adaptation)
- **任务类型**: 信息提取 (Information Extraction)
- **语言**: 中文 (Chinese)
- **许可证**: MIT

## 功能特点

- 🚀 基于最新的Qwen2.5-0.5B-Instruct模型
- 📦 专门针对物流填单场景优化
- 🔍 支持多种信息提取：收件人、地址、电话、寄件人等
- 🎯 高精度结构化输出
- 💾 轻量化设计，支持4bit量化

## 支持的信息类型

- **收件人信息**: 收件人姓名、收件人电话、收件地址
- **寄件人信息**: 寄件人姓名、寄件人电话、寄件地址
- **其他信息**: 公司名称、备注、重量等

## 使用方法

### 使用Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("your-username/logistics-extractor")
tokenizer = AutoTokenizer.from_pretrained("your-username/logistics-extractor")

# 准备输入
input_text = "收件人：张三，收件地址：北京市朝阳区中关村大街123号，收件人电话：13800138000"
prompt = f"<|im_start|>system\n你是一个专业的物流信息提取助手，能够从物流填单信息中准确提取结构化数据。<|im_end|>\n<|im_start|>user\n请从以下物流填单信息中提取结构化数据\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

# 生成回复
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### 使用我们的推理脚本

```bash
python inference.py --model_path your-username/logistics-extractor --input_text "收件人：张三，地址：北京市朝阳区xxx街道，电话：13800138000"
```

## 训练数据

模型使用自定义生成的物流填单数据进行训练，包含以下特点：

- **训练样本**: 5,000条
- **验证样本**: 500条
- **测试样本**: 500条
- **数据格式**: JSON格式的结构化数据
- **覆盖范围**: 多种物流填单格式和场景

## 模型性能

在测试集上的性能表现：

- **完全匹配准确率**: >95%
- **部分匹配准确率**: >98%
- **成功率**: >99%
- **推理速度**: ~2秒/样本 (CPU)

## 训练配置

- **基础模型**: Qwen2.5-0.5B-Instruct
- **微调方法**: LoRA
- **LoRA配置**: r=16, alpha=32
- **训练轮数**: 3 epochs
- **学习率**: 2e-4
- **批次大小**: 4
- **最大序列长度**: 2048

## 限制和注意事项

1. **语言限制**: 目前仅支持中文物流填单信息提取
2. **格式要求**: 输入文本需要包含明确的字段标识（如"收件人："）
3. **数据质量**: 模型性能依赖于输入文本的质量和格式
4. **计算资源**: 推理需要一定的计算资源，建议使用GPU加速

## 引用

如果您在研究中使用了这个模型，请引用：

```bibtex
@misc{logistics-extractor-2024,
  author = {您的名字},
  title = {物流填单信息提取模型},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/your-username/logistics-extractor}
}
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个模型！

## 联系方式

- GitHub: [您的GitHub链接]
- Email: [您的邮箱]

---

*此模型基于Qwen2.5-0.5B-Instruct微调，遵循原模型的许可证要求。* 