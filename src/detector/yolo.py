from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Dict, Union
from .config import DetectorConfig
from src.utils.logger import setup_logger


class ObjectDetector:
    """YOLO对象检测器类"""

    def __init__(self, config: DetectorConfig = None):
        """
        初始化YOLO检测器

        Args:
            config: 检测器配置，如果为None则使用默认配置
        """
        self.config = config or DetectorConfig()
        self.logger = setup_logger('detector')

        # 确保模型文件存在
        if not self.config.MODEL_PATH.exists():
            self.logger.warning(f"模型文件不存在: {self.config.MODEL_PATH}")
            self.logger.info("正在下载YOLOv8n模型...")
            self.config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 加载YOLO模型
            self.model = YOLO(self.config.MODEL_PATH)
            self.logger.info(f"成功加载模型: {self.config.MODEL_PATH}")
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Union[str, float, List[float]]]]:
        """
        对输入帧进行目标检测

        Args:
            frame: 输入图像帧 (BGR格式)

        Returns:
            检测到的目标列表，每个目标包含类别、置信度和边界框信息
        """
        try:
            # 运行推理
            results = self.model(
                frame,
                conf=self.config.CONFIDENCE_THRESHOLD,
                device=self.config.DEVICE
            )

            # 处理检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 获取类别信息
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]

                    # 如果不在目标类别中，跳过
                    if cls_name not in self.config.TARGET_CLASSES:
                        continue

                    # 获取置信度和边界框
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()  # 转换为列表格式

                    # 添加到检测结果
                    detection = {
                        'class': cls_name,
                        'confidence': confidence,
                        'bbox': xyxy  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)

            # 按置信度排序并限制检测数量
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            detections = detections[:self.config.MAX_DETECTIONS]

            return detections

        except Exception as e:
            self.logger.error(f"检测过程出错: {str(e)}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            frame: 输入图像帧
            detections: 检测结果列表

        Returns:
            绘制了检测框的图像帧
        """
        img = frame.copy()
        for det in detections:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, det['bbox'])

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 准备标签文本
            label = f"{det['class']} {det['confidence']:.2f}"

            # 计算标签大小
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 绘制标签背景
            cv2.rectangle(
                img,
                (x1, y1 - label_height - baseline),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )

            # 绘制标签文本
            cv2.putText(
                img, label, (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        return img

    def __call__(self, frame: np.ndarray) -> List[Dict]:
        """
        调用对象检测

        Args:
            frame: 输入图像帧

        Returns:
            检测结果列表
        """
        return self.detect(frame)