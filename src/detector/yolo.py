# src/detector/yolo.py
from ultralytics import YOLO
from pathlib import Path
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

        try:
            # 自动加载或下载模型
            self.model = YOLO(self.config.MODEL_PATH)
            self.logger.info(f"成功加载模型: {self.config.MODEL_PATH}")
        except Exception as e:
            self.logger.warning(f"尝试加载模型失败: {str(e)}")
            self.logger.info(f"尝试直接通过 ultralytics 自动下载模型 {self.config.MODEL_NAME}...")
            try:
                self.model = YOLO(self.config.MODEL_NAME)  # ultralytics 会自动下载模型
                self.logger.info(f"成功通过 ultralytics 下载并加载模型: {self.config.MODEL_NAME}")
            except Exception as ex:
                self.logger.error(f"自动下载并加载模型失败: {str(ex)}")
                raise RuntimeError(f"无法加载模型: {str(ex)}")

    @staticmethod
    def estimate_distance(bbox, frame_width, focal_length=500, real_object_width=0.5):
        """
        根据边界框估算目标与摄像头的距离。

        Args:
            bbox (List[float]): 边界框 [x1, y1, x2, y2]。
            frame_width (int): 图像帧的宽度。
            focal_length (float): 摄像头的焦距，默认值为 500。
            real_object_width (float): 目标物体的实际宽度（单位：米），默认值为 0.5。

        Returns:
            float: 估算的距离（单位：米）。
        """
        object_width_in_pixels = bbox[2] - bbox[0]
        if object_width_in_pixels == 0:
            return float('inf')  # 防止除零错误

        distance = (real_object_width * focal_length) / object_width_in_pixels
        return round(distance, 2)  # 保留两位小数

    def detect(self, frame: np.ndarray) -> List[Dict[str, Union[str, float, List[float]]]]:
        """
        对输入帧进行目标检测，并估算每个目标的距离。

        Args:
            frame: 输入图像帧 (BGR格式)。

        Returns:
            List[Dict[str, Union[str, float, List[float]]]]: 检测结果列表。
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

                    # 估算目标距离
                    distance = self.estimate_distance(
                        bbox=xyxy,
                        frame_width=frame.shape[1],  # 使用当前帧的宽度
                        focal_length=500,  # 摄像头焦距
                        real_object_width=0.5  # 目标实际宽度（根据目标类型调整）
                    )

                    # 添加到检测结果
                    detection = {
                        'class': cls_name,
                        'confidence': confidence,
                        'bbox': xyxy,  # [x1, y1, x2, y2]
                        'distance': distance  # 距离（单位：米）
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