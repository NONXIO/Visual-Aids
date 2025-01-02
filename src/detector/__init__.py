# src/camera/__init__.py
from .yolo import ObjectDetector
from .config import DetectorConfig

__all__ = ['ObjectDetector', 'DetectorConfig']