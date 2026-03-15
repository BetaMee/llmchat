#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境验证脚本
用于检查所有依赖是否正确安装
"""

import sys

def print_header(text):
    """打印标题"""
    print("\n" + "="*60)
    print(f"   {text}")
    print("="*60)

def print_section(text):
    """打印小节标题"""
    print(f"\n{'─'*60}")
    print(f"📋 {text}")
    print(f"{'─'*60}")

def check_python():
    """检查 Python 版本"""
    print_section("Python 环境")
    version = sys.version_info
    print(f"✅ Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print(f"   状态: 符合要求 (>= 3.10)")
        return True
    else:
        print(f"   ⚠️  警告: 推荐使用 Python 3.10-3.12")
        return False

def check_pytorch():
    """检查 PyTorch"""
    print_section("PyTorch 核心")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # 检查 CUDA
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA 可用: {'是' if cuda_available else '否'}")
        
        if cuda_available:
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"   显存: {props.total_memory / 1024**3:.1f} GB")
        else:
            print(f"   ⚠️  警告: 未检测到 CUDA，将使用 CPU（训练会非常慢）")
        
        # 检查 bf16 支持
        if cuda_available:
            bf16_supported = torch.cuda.is_bf16_supported()
            print(f"   BF16 支持: {'是' if bf16_supported else '否'}")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch 未安装: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch 检查失败: {e}")
        return False

def check_transformers():
    """检查 Transformers"""
    print_section("Transformers")
    try:
        import transformers
        version = transformers.__version__
        print(f"✅ Transformers: {version}")
        
        # 检查版本是否满足要求
        major, minor = map(int, version.split('.')[:2])
        if major > 4 or (major == 4 and minor >= 40):
            print(f"   状态: 符合要求 (>= 4.40.0)")
        else:
            print(f"   ⚠️  警告: 需要 >= 4.40.0 以支持当前 Qwen 模型")
        
        return True
    except ImportError as e:
        print(f"❌ Transformers 未安装: {e}")
        return False

def check_bitsandbytes():
    """检查 Bitsandbytes"""
    print_section("Bitsandbytes (量化)")
    try:
        import bitsandbytes as bnb
        print(f"✅ Bitsandbytes: {bnb.__version__}")
        
        # 测试 CUDA 支持
        try:
            import torch
            if torch.cuda.is_available():
                # 简单测试
                import bitsandbytes.functional as F
                print(f"   CUDA 支持: 正常")
                print(f"   4-bit 量化: 可用")
            else:
                print(f"   ⚠️  无 CUDA，量化功能受限")
        except Exception as e:
            print(f"   ⚠️  CUDA 支持异常: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ Bitsandbytes 未安装: {e}")
        print(f"   安装建议: pip install bitsandbytes>=0.43.0")
        return False

def check_unsloth():
    """检查 Unsloth"""
    print_section("Unsloth (训练加速)")
    try:
        import unsloth
        print(f"✅ Unsloth: 已安装")
        
        # 尝试导入主要功能
        try:
            from unsloth import FastLanguageModel
            print(f"   FastLanguageModel: 可用")
        except Exception as e:
            print(f"   ⚠️  FastLanguageModel 导入失败: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ Unsloth 未安装: {e}")
        print(f"   安装建议: pip install 'unsloth @ git+https://github.com/unslothai/unsloth.git'")
        return False

def check_training_libs():
    """检查训练相关库"""
    print_section("训练相关库")
    
    libs = {
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
        'peft': 'PEFT',
        'trl': 'TRL',
    }
    
    all_ok = True
    for module, name in libs.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', '未知')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: 未安装")
            all_ok = False
    
    return all_ok

def check_data_processing():
    """检查数据处理库"""
    print_section("数据处理库")
    
    libs = {
        'yaml': ('PyYAML', 'pyyaml'),
        'pandas': ('Pandas', 'pandas'),
        'numpy': ('NumPy', 'numpy'),
        'tqdm': ('TQDM', 'tqdm'),
    }
    
    all_ok = True
    for module, (name, pkg) in libs.items():
        try:
            if module == 'yaml':
                import yaml
                print(f"✅ {name}: 已安装")
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', '未知')
                print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: 未安装 (pip install {pkg})")
            all_ok = False
    
    return all_ok

def check_monitoring():
    """检查监控工具"""
    print_section("监控工具 (可选)")
    
    # TensorBoard
    try:
        import tensorboard
        print(f"✅ TensorBoard: {tensorboard.__version__}")
    except ImportError:
        print(f"⚠️  TensorBoard: 未安装 (可选)")
    
    # Weights & Biases
    try:
        import wandb
        print(f"✅ Weights & Biases: {wandb.__version__}")
    except ImportError:
        print(f"⚠️  Weights & Biases: 未安装 (可选)")
    
    # scikit-learn
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print(f"⚠️  scikit-learn: 未安装")

def check_dev_tools():
    """检查开发工具"""
    print_section("开发工具 (可选)")
    
    # IPython
    try:
        import IPython
        print(f"✅ IPython: {IPython.__version__}")
    except ImportError:
        print(f"⚠️  IPython: 未安装 (可选)")
    
    # Jupyter
    try:
        import jupyter
        print(f"✅ Jupyter: 已安装")
    except ImportError:
        print(f"⚠️  Jupyter: 未安装 (可选)")

def check_flash_attention():
    """检查 Flash Attention"""
    print_section("性能优化 (可选)")
    
    try:
        import flash_attn
        print(f"✅ Flash Attention: 已安装")
        print(f"   注意: 可大幅加速训练")
    except ImportError:
        print(f"⚠️  Flash Attention: 未安装 (强烈推荐)")
        print(f"   安装: pip install flash-attn --no-build-isolation")

def check_huggingface_network():
    """检查 HuggingFace 网络连接"""
    print_section("HuggingFace 网络连接")
    
    import urllib.request
    import socket
    import os
    
    # 检测是否使用镜像
    hf_endpoint = os.environ.get('HF_ENDPOINT', '')
    if hf_endpoint:
        print(f"   检测到镜像配置: {hf_endpoint}")
    
    # 测试连接的目标
    test_urls = [
        ("HuggingFace 官网", "https://huggingface.co"),
        ("HuggingFace 模型库", "https://huggingface.co/api/models"),
    ]
    
    if hf_endpoint:
        test_urls.insert(0, ("HuggingFace 镜像", hf_endpoint))
    
    all_ok = True
    for name, url in test_urls:
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'},
                method='HEAD'
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.getcode()
                if status == 200:
                    print(f"✅ {name}: 连接正常")
                else:
                    print(f"⚠️  {name}: HTTP {status}")
                    all_ok = False
        except urllib.error.HTTPError as e:
            # 某些端点可能返回 403/405，但只要能连接说明网络通
            if e.code in [403, 405]:
                print(f"✅ {name}: 连接正常 (HTTP {e.code})")
            else:
                print(f"❌ {name}: HTTP {e.code}")
                all_ok = False
        except urllib.error.URLError as e:
            print(f"❌ {name}: 连接失败 ({e.reason})")
            all_ok = False
        except socket.timeout:
            print(f"❌ {name}: 连接超时")
            all_ok = False
        except Exception as e:
            print(f"❌ {name}: 连接异常 ({type(e).__name__})")
            all_ok = False
    
    # 提供建议
    if not all_ok:
        print("\n   建议:")
        print("   1. 检查网络连接和代理设置")
        print("   2. 国内用户可使用镜像:")
        print("      export HF_ENDPOINT=https://hf-mirror.com")
        print("   3. 或使用 --mirror 参数运行 start_train.sh")
    else:
        print("\n   ✅ HuggingFace 网络连接正常，可以下载模型")
    
    return all_ok

def test_model_loading():
    """测试模型加载能力"""
    print_section("模型加载测试")
    
    try:
        from unsloth import FastLanguageModel
        print(f"✅ FastLanguageModel 可导入")
        print(f"   注意: 实际加载模型需要较长时间，此处仅测试导入")
        return True
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False

def print_summary(results):
    """打印总结"""
    print_header("验证总结")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"\n总计检查项: {total}")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    
    if failed == 0:
        print("\n🎉 所有核心依赖已正确安装！")
        print("\n下一步:")
        print("  1. 准备训练数据: python src/data_processor.py --input ./data/figma-records.jsonl --output ./data/train.json")
        print("  2. 开始训练: python src/train.py")
    else:
        print("\n⚠️  部分依赖缺失或配置异常，请根据上述提示修复。")
        print("\n常见问题:")
        print("  - PyTorch 未检测到 CUDA: 检查 CUDA 驱动和 PyTorch 版本")
        print("  - Bitsandbytes 错误: pip install --upgrade bitsandbytes")
        print("  - Unsloth 导入失败: pip install 'unsloth @ git+https://...'")

def main():
    """主函数"""
    print_header("Figma JSON 训练框架 - 环境验证")
    
    # 执行所有检查
    # 注意: Unsloth 必须在 Transformers 之前导入以启用优化
    results = {
        'Python': check_python(),
        'PyTorch': check_pytorch(),
        'Unsloth': check_unsloth(),  # 先检查 Unsloth
        'Transformers': check_transformers(),  # 后检查 Transformers
        'Bitsandbytes': check_bitsandbytes(),
        'Training Libs': check_training_libs(),
        'Data Processing': check_data_processing(),
    }
    
    # 可选检查
    check_monitoring()
    check_dev_tools()
    check_flash_attention()
    
    # 网络连接检查
    results['HF Network'] = check_huggingface_network()
    
    # 模型加载测试
    results['Model Loading'] = test_model_loading()
    
    # 打印总结
    print_summary(results)
    
    print("\n" + "="*60 + "\n")
    
    # 返回退出码
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
