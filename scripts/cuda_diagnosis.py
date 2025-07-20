#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA 详细诊断脚本
专门用于诊断 RTX 4070 CUDA 不可用的问题
"""

import sys
import subprocess
import os
from typing import Dict, Any


def check_nvidia_driver():
    """检查 NVIDIA 驱动"""
    print("=" * 60)
    print("1. NVIDIA 驱动检查")
    print("=" * 60)
    
    try:
        # 检查 nvidia-smi 命令
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ nvidia-smi 命令可用")
            print("驱动信息:")
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # 只显示前10行
                if line.strip():
                    print(f"  {line}")
        else:
            print("❌ nvidia-smi 命令失败")
            print(f"错误输出: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi 命令未找到")
        print("  请安装 NVIDIA 驱动")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi 命令超时")
        return False
    
    return True


def check_cuda_toolkit():
    """检查 CUDA Toolkit"""
    print("\n" + "=" * 60)
    print("2. CUDA Toolkit 检查")
    print("=" * 60)
    
    # 检查环境变量
    cuda_path = os.environ.get('CUDA_PATH')
    cuda_home = os.environ.get('CUDA_HOME')
    
    print(f"CUDA_PATH: {cuda_path or '未设置'}")
    print(f"CUDA_HOME: {cuda_home or '未设置'}")
    
    # 检查 nvcc 命令
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ nvcc 命令可用")
            lines = result.stdout.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"  {line}")
        else:
            print("❌ nvcc 命令失败")
            print(f"错误输出: {result.stderr}")
    except FileNotFoundError:
        print("❌ nvcc 命令未找到")
        print("  请安装 CUDA Toolkit")
    
    # 检查常见的 CUDA 安装路径
    common_cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",
        r"C:\ProgramData\NVIDIA GPU Computing Toolkit\CUDA"
    ]
    
    print("\n检查常见 CUDA 安装路径:")
    for path in common_cuda_paths:
        if os.path.exists(path):
            print(f"  ✓ 找到: {path}")
            # 列出子目录
            try:
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                for subdir in subdirs:
                    print(f"    - {subdir}")
            except:
                pass
        else:
            print(f"  ❌ 未找到: {path}")


def check_pytorch_installation():
    """检查 PyTorch 安装"""
    print("\n" + "=" * 60)
    print("3. PyTorch 安装检查")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        
        # 检查是否安装了 CUDA 版本
        if '+cu' in torch.__version__:
            print("✓ PyTorch 包含 CUDA 支持")
            cuda_version = torch.version.cuda
            print(f"编译时的 CUDA 版本: {cuda_version}")
        else:
            print("❌ PyTorch 不包含 CUDA 支持")
            print("  这是 CPU 版本的 PyTorch")
        
        # 检查 CUDA 可用性
        print(f"\nCUDA 可用性检查:")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  ❌ CUDA 不可用")
            
            # 尝试获取更详细的错误信息
            try:
                # 强制尝试 CUDA 操作
                device = torch.device("cuda:0")
                x = torch.tensor([1.0]).to(device)
                print("  ✓ CUDA 操作测试成功")
            except Exception as e:
                print(f"  ❌ CUDA 操作测试失败: {e}")
        
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    return True


def check_system_info():
    """检查系统信息"""
    print("\n" + "=" * 60)
    print("4. 系统信息")
    print("=" * 60)
    
    import platform
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.architecture()}")
    print(f"机器: {platform.machine()}")
    
    # 检查 Windows 版本
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            version = winreg.QueryValueEx(key, "ProductName")[0]
            print(f"Windows 版本: {version}")
        except:
            pass


def check_environment_variables():
    """检查环境变量"""
    print("\n" + "=" * 60)
    print("5. 环境变量检查")
    print("=" * 60)
    
    cuda_vars = [
        'CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT',
        'PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'
    ]
    
    for var in cuda_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var}: 未设置")


def check_python_packages():
    """检查 Python 包"""
    print("\n" + "=" * 60)
    print("6. Python 包检查")
    print("=" * 60)
    
    packages = [
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'peft', 'accelerate',
        'bitsandbytes', 'datasets'
    ]
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: 未安装")


def provide_solutions():
    """提供解决方案"""
    print("\n" + "=" * 60)
    print("7. 解决方案建议")
    print("=" * 60)
    
    print("如果 CUDA 不可用，请按以下步骤解决：")
    print()
    
    print("1. 安装 NVIDIA 驱动:")
    print("   - 访问 https://www.nvidia.com/Download/index.aspx")
    print("   - 选择 RTX 4070 和你的操作系统")
    print("   - 下载并安装最新驱动")
    print()
    
    print("2. 安装 CUDA Toolkit:")
    print("   - 访问 https://developer.nvidia.com/cuda-downloads")
    print("   - 选择适合你系统的版本")
    print("   - 建议安装 CUDA 11.8 或 12.1")
    print()
    
    print("3. 重新安装支持 CUDA 的 PyTorch:")
    print("   - 卸载当前 PyTorch: pip uninstall torch torchvision torchaudio")
    print("   - 安装 CUDA 版本:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   - 或者使用 conda:")
    print("     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print()
    
    print("4. 验证安装:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")
    print()


def main():
    """主函数"""
    print("RTX 4070 CUDA 诊断脚本")
    print("=" * 60)
    
    # 运行各项检查
    driver_ok = check_nvidia_driver()
    check_cuda_toolkit()
    pytorch_ok = check_pytorch_installation()
    check_system_info()
    check_environment_variables()
    check_python_packages()
    
    # 提供解决方案
    provide_solutions()
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main() 