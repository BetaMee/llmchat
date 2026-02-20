#!/bin/bash

# Figma JSON 训练框架快速设置脚本

echo "==================================="
echo "Figma JSON 训练框架 - 快速设置"
echo "==================================="

# 检查 Python 版本
echo ""
echo "检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 创建必要的目录
echo ""
echo "创建项目目录..."
mkdir -p data
mkdir -p outputs
mkdir -p logs

# 创建 .gitkeep 文件
touch data/.gitkeep

echo "✓ 目录创建完成"

# 询问是否安装依赖
echo ""
read -p "是否安装 Python 依赖？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "安装依赖中..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✓ 依赖安装完成"
    else
        echo "✗ 依赖安装失败，请手动安装"
    fi
fi

echo ""
echo "==================================="
echo "设置完成！"
echo "==================================="
echo ""
echo "下一步："
echo "1. 检查 config.yaml 配置"
echo "2. 准备训练数据（或使用生成的示例数据）"
echo "3. 运行训练: python3 train.py"
echo "4. 测试推理: python3 inference.py --mode test"
echo ""
echo "详细使用说明请查看 README.md"
echo ""
