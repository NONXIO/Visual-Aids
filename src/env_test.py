import os
import sys
import time


def test_import(package_name, actual_import=None):
    """测试导入包"""
    if actual_import is None:
        actual_import = package_name

    try:
        exec(f"import {actual_import}")
        print(f"✅ {package_name} 导入成功")
        return True
    except ImportError as e:
        print(f"❌ {package_name} 导入失败: {str(e)}")
        return False


def test_numpy():
    """测试NumPy基本功能"""
    import numpy as np

    # 创建并操作数组
    arr = np.array([1, 2, 3, 4, 5])
    result = np.mean(arr)
    print(f"✅ NumPy功能测试: 数组[1,2,3,4,5]的平均值 = {result}")


def test_opencv():
    """测试OpenCV基本功能"""
    import cv2
    import numpy as np

    # 创建简单图像并显示版本
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)  # 绿色图像
    print(f"✅ OpenCV功能测试: 版本 {cv2.__version__}，成功创建图像")


def test_pandas():
    """测试Pandas基本功能"""
    import pandas as pd

    # 创建并操作DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"✅ Pandas功能测试: 成功创建DataFrame\n{df.head()}")


def test_matplotlib():
    """测试Matplotlib基本功能"""
    import matplotlib
    matplotlib.use('Agg')  # 非交互模式
    import matplotlib.pyplot as plt

    # 创建简单图表
    plt.figure(figsize=(2, 2))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    print(f"✅ Matplotlib功能测试: 版本 {matplotlib.__version__}，成功创建图表")


def test_pytorch():
    """测试PyTorch基本功能"""
    import torch

    # 创建并操作张量
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    z = x + y

    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "N/A"

    print(f"✅ PyTorch功能测试: 版本 {torch.__version__}")
    print(f"   - CUDA可用: {cuda_available}")
    print(f"   - CUDA设备数量: {device_count}")
    print(f"   - CUDA设备名称: {cuda_device}")
    print(f"   - 张量操作: {x} + {y} = {z}")


def test_ultralytics():
    """测试Ultralytics基本功能"""
    from ultralytics import YOLO

    # 只检查版本，不加载模型
    import ultralytics
    print(f"✅ Ultralytics功能测试: 版本 {ultralytics.__version__}")


def test_gtts():
    """测试gTTS基本功能"""
    from gtts import gTTS

    # 创建一个简单的音频测试（不保存）
    try:
        tts = gTTS("测试文本转语音功能", lang="zh")
        print(f"✅ gTTS功能测试: 成功初始化TTS引擎")
    except Exception as e:
        print(f"❌ gTTS功能测试失败: {str(e)}")


def test_paddleocr():
    """测试PaddleOCR基本功能"""
    try:
        import paddleocr
        # 只检查包是否可以导入，不初始化OCR引擎
        print(f"✅ PaddleOCR功能测试: 版本 {paddleocr.__version__}")
    except Exception as e:
        print(f"❌ PaddleOCR功能测试失败: {str(e)}")


def main():
    """执行所有测试"""
    print("\n===== Visual-Aids 环境测试 =====\n")

    # 测试Python版本
    python_version = sys.version
    print(f"Python版本: {python_version}")

    # 测试各包导入
    packages = [
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pytorch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("gTTS", "gtts"),
        ("paddleocr", "paddleocr"),
        ("paddlepaddle", "paddle")
    ]

    import_results = {}
    for package, import_name in packages:
        result = test_import(package, import_name)
        import_results[package] = result

    print("\n===== 功能测试 =====\n")

    # 测试各包功能
    if import_results.get("numpy", False):
        test_numpy()

    if import_results.get("opencv-python", False):
        test_opencv()

    if import_results.get("pandas", False):
        test_pandas()

    if import_results.get("matplotlib", False):
        test_matplotlib()

    if import_results.get("pytorch", False):
        test_pytorch()

    if import_results.get("ultralytics", False):
        test_ultralytics()

    if import_results.get("gTTS", False):
        test_gtts()

    if import_results.get("paddleocr", False) and import_results.get("paddlepaddle", False):
        test_paddleocr()

    print("\n===== 测试完成 =====\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"测试耗时: {end_time - start_time:.2f} 秒")