# 物流填单信息提取模型

基于Qwen3-0.6B微调的物流填单信息结构化提取模型，能够从物流填单文本中提取收件人、地址、电话等关键信息。

## 项目特点

- 🚀 基于Qwen3-0.6B大语言模型微调
- 📦 专门针对物流填单场景优化
- 🔍 支持多种信息提取：收件人、地址、电话、寄件人等
- 🎯 高精度结构化输出
- 📊 完整的训练和评估流程

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 模型训练

```bash
python train.py --config configs/train_config.yaml
```

### 模型推理

```bash
python inference.py --model_path ./outputs/checkpoint-final --input_text "收件人：张三，地址：北京市朝阳区xxx街道，电话：13800138000"
```

## 项目结构

```
├── configs/                 # 配置文件
├── data/                    # 数据目录
├── src/                     # 源代码
│   ├── data_processing/     # 数据处理
│   ├── model/              # 模型定义
│   ├── training/           # 训练相关
│   └── utils/              # 工具函数
├── scripts/                # 脚本文件
├── outputs/                # 输出目录
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 数据格式

训练数据采用JSON格式：

```json
{
  "instruction": "请从以下物流填单信息中提取结构化数据",
  "input": "收件人：张三，地址：北京市朝阳区xxx街道123号，电话：13800138000",
  "output": "{\"收件人\": \"张三\", \"地址\": \"北京市朝阳区xxx街道123号\", \"电话\": \"13800138000\"}"
}
```

## 模型性能

- 准确率：>95%
- 召回率：>92%
- 支持多种物流填单格式

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
