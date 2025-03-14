# src/detector/config.py
from pathlib import Path
import torch

class DetectorConfig:
    """YOLO检测器配置"""

    # 模型配置
    MODEL_NAME = "yolo11l"  # 使用yolo11l作为默认模型
    MODEL_PATH = Path(f"../../models/{MODEL_NAME}.pt")  # 模型文件路径
    CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值

    # 推理设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择
    INPUT_WIDTH = 640  # 输入图像宽度
    INPUT_HEIGHT = 640  # 输入图像高度

    MIN_DISTANCE = 0.25  # 最小检测距离（单位：米）
    MAX_DISTANCE = 10.0  # 最大检测距离（单位：米）

    # 目标配置 - 主要关注户外场景中的重要物体
    TARGET_CLASSES = [
       'bus','person','bicycle','car','motorbike','truck',
        'traffic light','stop sign'
    ]

    # 添加物体实际宽度配置
    OBJECT_REAL_WIDTHS = {
        'person': 0.45,  # 平均肩宽
        'bus': 2.5,
        'car': 1.8,
        'bicycle': 0.6,
        'motorbike': 0.7,
        'truck': 2.5
    }

    # 处理配置
    BATCH_SIZE = 1  # 实时处理用单批次
    MAX_DETECTIONS = 20  # 每帧最大检测数量
