# src/camera/__init__.py
from .yolo import ObjectDetector
from .yolo_config import DetectorConfig

__all__ = ['ObjectDetector', 'DetectorConfig']