# src/detector/config.py
from pathlib import Path
import os


class DetectorConfig:
    """YOLO检测器配置"""

    # 模型配置
    MODEL_PATH = Path("../../models/yolov8n.pt")  # 使用YOLOv8n作为默认模型
    CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值

    # 推理配置
    DEVICE = "cpu"  # 使用CPU进行推理
    INPUT_WIDTH = 640  # 输入图像宽度
    INPUT_HEIGHT = 640  # 输入图像高度

    # 目标配置 - 主要关注户外场景中的重要物体
    TARGET_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
        'traffic light', 'stop sign', 'bench', 'dog', 'cat'
    ]

    # 处理配置
    BATCH_SIZE = 1  # 实时处理用单批次
    MAX_DETECTIONS = 20  # 每帧最大检测数量