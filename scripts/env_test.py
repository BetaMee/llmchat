#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA 环境测试脚本
用于检查当前环境是否支持 CUDA 以及相关库的可用性
"""

import sys
import platform
from typing import Dict, Any


def test_python_version() -> Dict[str, Any]:
    """测试 Python 版本"""
    print("=" * 50)
    print("Python 版本信息")
    print("=" * 50)
    
    version_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "machine": platform.machine()
    }
    
    print(f"Python 版本: {version_info['python_version']}")
    print(f"平台: {version_info['platform']}")
    print(f"架构: {version_info['architecture']}")
    print(f"机器类型: {version_info['machine']}")
    
    return version_info


def test_torch_cuda() -> Dict[str, Any]:
    """测试 PyTorch CUDA 支持"""
    print("\n" + "=" * 50)
    print("PyTorch CUDA 支持测试")
    print("=" * 50)
    
    try:
        import torch
        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        print(f"PyTorch 版本: {torch_info['torch_version']}")
        print(f"CUDA 可用: {torch_info['cuda_available']}")
        
        if torch_info['cuda_available']:
            print(f"CUDA 版本: {torch_info['cuda_version']}")
            print(f"cuDNN 版本: {torch_info['cudnn_version']}")
            print(f"GPU 数量: {torch_info['device_count']}")
            
            # 显示每个 GPU 的信息
            for i in range(torch_info['device_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
            # 测试 CUDA 张量操作
            print("\n测试 CUDA 张量操作...")
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print("✓ CUDA 张量操作成功")
            
        else:
            print("❌ CUDA 不可用")
            
        return torch_info
        
    except ImportError:
        print("❌ PyTorch 未安装")
        return {"error": "PyTorch not installed"}


def test_transformers() -> Dict[str, Any]:
    """测试 Transformers 库"""
    print("\n" + "=" * 50)
    print("Transformers 库测试")
    print("=" * 50)
    
    try:
        import transformers
        transformers_info = {
            "transformers_version": transformers.__version__,
            "tokenizers_version": getattr(transformers, '__tokenizers_version__', 'N/A')
        }
        
        print(f"Transformers 版本: {transformers_info['transformers_version']}")
        print(f"Tokenizers 版本: {transformers_info['tokenizers_version']}")
        
        # 测试模型加载
        print("\n测试模型加载...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        print("✓ Tokenizer 加载成功")
        
        return transformers_info
        
    except ImportError:
        print("❌ Transformers 未安装")
        return {"error": "Transformers not installed"}
    except Exception as e:
        print(f"❌ Transformers 测试失败: {e}")
        return {"error": str(e)}


def test_peft() -> Dict[str, Any]:
    """测试 PEFT 库"""
    print("\n" + "=" * 50)
    print("PEFT 库测试")
    print("=" * 50)
    
    try:
        import peft
        peft_info = {
            "peft_version": peft.__version__
        }
        
        print(f"PEFT 版本: {peft_info['peft_version']}")
        print("✓ PEFT 库可用")
        
        return peft_info
        
    except ImportError:
        print("❌ PEFT 未安装")
        return {"error": "PEFT not installed"}


def test_bitsandbytes() -> Dict[str, Any]:
    """测试 bitsandbytes 库"""
    print("\n" + "=" * 50)
    print("bitsandbytes 库测试")
    print("=" * 50)
    
    try:
        import bitsandbytes as bnb
        bnb_info = {
            "bitsandbytes_version": bnb.__version__
        }
        
        print(f"bitsandbytes 版本: {bnb_info['bitsandbytes_version']}")
        
        # 尝试不同的方法来检查 CUDA 支持
        cuda_available = False
        try:
            # 方法1：尝试导入 CUDA 相关模块
            from bitsandbytes.cuda_setup.main import get_compute_capability
            compute_cap = get_compute_capability()
            cuda_available = compute_cap is not None
            print(f"CUDA 支持: {cuda_available}")
            if cuda_available:
                print(f"计算能力: {compute_cap}")
        except ImportError:
            try:
                # 方法2：尝试直接访问
                compute_cap = bnb.cuda_setup.get_compute_capability()
                cuda_available = compute_cap is not None
                print(f"CUDA 支持: {cuda_available}")
                if cuda_available:
                    print(f"计算能力: {compute_cap}")
            except AttributeError:
                try:
                    # 方法3：尝试其他属性
                    cuda_available = hasattr(bnb, 'cuda_setup')
                    print(f"CUDA 支持: {cuda_available}")
                except:
                    print("无法确定 CUDA 支持状态")
        
        bnb_info["cuda_available"] = cuda_available
        
        if cuda_available:
            print("✓ bitsandbytes CUDA 支持可用")
        else:
            print("⚠ bitsandbytes 没有 CUDA 支持")
            
        return bnb_info
        
    except ImportError:
        print("❌ bitsandbytes 未安装")
        return {"error": "bitsandbytes not installed"}


def test_accelerate() -> Dict[str, Any]:
    """测试 Accelerate 库"""
    print("\n" + "=" * 50)
    print("Accelerate 库测试")
    print("=" * 50)
    
    try:
        import accelerate
        accelerate_info = {
            "accelerate_version": accelerate.__version__
        }
        
        print(f"Accelerate 版本: {accelerate_info['accelerate_version']}")
        
        # 测试 Accelerate 配置
        from accelerate import Accelerator
        accelerator = Accelerator()
        print("✓ Accelerate 初始化成功")
        
        return accelerate_info
        
    except ImportError:
        print("❌ Accelerate 未安装")
        return {"error": "Accelerate not installed"}


def test_datasets() -> Dict[str, Any]:
    """测试 Datasets 库"""
    print("\n" + "=" * 50)
    print("Datasets 库测试")
    print("=" * 50)
    
    try:
        import datasets
        datasets_info = {
            "datasets_version": datasets.__version__
        }
        
        print(f"Datasets 版本: {datasets_info['datasets_version']}")
        print("✓ Datasets 库可用")
        
        return datasets_info
        
    except ImportError:
        print("❌ Datasets 未安装")
        return {"error": "Datasets not installed"}


def generate_summary(all_results: Dict[str, Any]):
    """生成测试总结"""
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    # 检查关键组件
    torch_ok = all_results.get('torch', {}).get('cuda_available', False)
    transformers_ok = 'error' not in all_results.get('transformers', {})
    peft_ok = 'error' not in all_results.get('peft', {})
    datasets_ok = 'error' not in all_results.get('datasets', {})
    
    print(f"PyTorch CUDA 支持: {'✓' if torch_ok else '❌'}")
    print(f"Transformers: {'✓' if transformers_ok else '❌'}")
    print(f"PEFT: {'✓' if peft_ok else '❌'}")
    print(f"Datasets: {'✓' if datasets_ok else '❌'}")
    
    if torch_ok and transformers_ok and peft_ok and datasets_ok:
        print("\n🎉 环境配置完整，可以进行模型训练！")
    else:
        print("\n⚠️  环境配置不完整，请检查缺失的组件。")
        
        if not torch_ok:
            print("  - 需要安装支持 CUDA 的 PyTorch")
        if not transformers_ok:
            print("  - 需要安装 Transformers")
        if not peft_ok:
            print("  - 需要安装 PEFT")
        if not datasets_ok:
            print("  - 需要安装 Datasets")


def main():
    """主函数"""
    print("CUDA 环境测试脚本")
    print("=" * 50)
    
    all_results = {}
    
    # 运行各项测试
    all_results['python'] = test_python_version()
    all_results['torch'] = test_torch_cuda()
    all_results['transformers'] = test_transformers()
    all_results['peft'] = test_peft()
    all_results['bitsandbytes'] = test_bitsandbytes()
    all_results['accelerate'] = test_accelerate()
    all_results['datasets'] = test_datasets()
    
    # 生成总结
    generate_summary(all_results)
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == "__main__":
    main() 