# src/detector/yolo.py
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Dict, Union
from .yolo_config import DetectorConfig
from src.utils.logger import setup_logger
import time


class ObjectDetector:
    """YOLO对象检测器类"""

    def __init__(self, config: DetectorConfig = None):
        """
        初始化YOLO检测器

        Args:
            config: 检测器配置，如果为None则使用默认配置
        """
        self.config = config or DetectorConfig()  # 如果未提供配置，则使用默认配置
        self.logger = setup_logger('detector')  # 设置日志记录器

        try:
            # 尝试加载本地模型
            self.model = YOLO(self.config.MODEL_PATH)
            self.logger.info(f"成功加载模型: {self.config.MODEL_PATH}")
        except Exception as e:
            # 如果本地加载失败，尝试在线下载模型
            self.logger.warning(f"尝试加载模型失败: {str(e)}")
            self.logger.info(f"尝试直接通过 ultralytics 自动下载模型 {self.config.MODEL_NAME}...")
            try:
                self.model = YOLO(self.config.MODEL_NAME)  # 自动下载并加载模型
                self.logger.info(f"成功通过 ultralytics 下载并加载模型: {self.config.MODEL_NAME}")
            except Exception as ex:
                # 如果在线下载也失败，则抛出错误
                self.logger.error(f"无法加载模型: {str(ex)}, 类型: {type(ex).__name__}")
                raise RuntimeError(f"无法加载模型: {str(ex)}")

    @staticmethod
    def estimate_distance(bbox, focal_length=500, cls_name=None, frame_width=None):
        """
        根据边界框估算目标与摄像头的距离。

        使用针孔相机模型公式：距离 = (物体实际宽度 * 焦距) / 物体像素宽度
        可根据图像分辨率动态调整焦距，提高距离估算准确性。

        Args:
            bbox (List[float]): 边界框 [x1, y1, x2, y2]。
            focal_length (float): 基准焦距，默认为500。
            cls_name (str): 目标物体类别，用于查找实际宽度。
            frame_width (int, optional): 图像帧的宽度，用于焦距调整。

        Returns:
            float: 估算的距离（单位：米），保留两位小数。
        """
        # 计算物体在图像中的宽度（像素）
        object_width_in_pixels = bbox[2] - bbox[0]

        # 防止除零错误
        if object_width_in_pixels <= 0:
            return float('inf')

        # 获取物体的实际宽度（如果类别未知，使用默认值0.5米）
        real_object_width = DetectorConfig.OBJECT_REAL_WIDTHS.get(cls_name, 0.5)

        # 根据图像宽度调整焦距（假设标准宽度为640像素）
        adjusted_focal_length = focal_length
        if frame_width is not None:
            standard_width = 640
            adjusted_focal_length = focal_length * (frame_width / standard_width)

        # 使用针孔相机模型计算距离
        distance = (real_object_width * adjusted_focal_length) / object_width_in_pixels

        # 应用非线性校正（可选，适用于近距离和远距离）
        # 这里使用简单的校正因子，实际应用中可能需要更复杂的校正
        correction_factor = 1.0
        if distance < 1.0:
            # 近距离校正（减少低估）
            correction_factor = 1.1
        elif distance > 5.0:
            # 远距离校正（减少高估）
            correction_factor = 0.9

        corrected_distance = distance * correction_factor

        # 确保距离在合理范围内
        min_distance = DetectorConfig.MIN_DISTANCE
        max_distance = DetectorConfig.MAX_DISTANCE
        clamped_distance = max(min_distance, min(corrected_distance, max_distance))

        # 返回保留两位小数的距离
        return round(clamped_distance, 2)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Union[str, float, List[float]]]]:
        """
        对输入帧进行目标检测，并估算每个目标的距离。

        Args:
            frame: 输入图像帧 (BGR格式)。

        Returns:
            List[Dict[str, Union[str, float, List[float]]]]: 检测结果列表。
        """
        try:
            # 将单帧复制为批量帧
            frames = [frame] * self.config.BATCH_SIZE
            self.logger.info(f"推理设备: {self.config.DEVICE}")
            start_time = time.time()  # 记录推理开始时间

            # 执行推理
            results = self.model(
                frames,  # 输入批量帧
                conf=self.config.CONFIDENCE_THRESHOLD,  # 置信度阈值
                device=self.config.DEVICE  # 推理设备
            )
            end_time = time.time()  # 记录推理结束时间
            self.logger.info(f"YOLO 推理耗时: {end_time - start_time:.3f} 秒")

            detections = []  # 存储检测结果
            for result in results:
                boxes = result.boxes  # 获取检测框信息
                for box in boxes:
                    cls_id = int(box.cls[0])  # 获取类别ID
                    cls_name = result.names[cls_id]  # 根据ID获取类别名称

                    # 如果检测的类别不在目标类别列表中，则跳过
                    if cls_name not in self.config.TARGET_CLASSES:
                        continue

                    confidence = float(box.conf[0])  # 获取置信度
                    xyxy = box.xyxy[0].tolist()  # 获取边界框坐标并转换为列表

                    # 估算物体距离
                    distance = self.estimate_distance(
                        bbox=xyxy,
                        frame_width=frame.shape[1],  # 图像宽度
                        cls_name=cls_name  # 物体类别
                    )

                    # 如果物体距离不在有效范围内，跳过
                    if not (self.config.MIN_DISTANCE <= distance <= self.config.MAX_DISTANCE):
                        self.logger.debug(f"物体 '{cls_name}' 被过滤，距离: {distance} 米")
                        continue

                    # 保存检测结果
                    detection = {
                        'class': cls_name,
                        'confidence': confidence,
                        'bbox': xyxy,  # [x1, y1, x2, y2]
                        'distance': distance  # 距离（单位：米）
                    }
                    detections.append(detection)

            # 过滤并排序检测结果
            return sorted(
                [det for det in detections if det['confidence'] >= self.config.CONFIDENCE_THRESHOLD],
                key=lambda x: x['confidence'],
                reverse=True
            )[:self.config.MAX_DETECTIONS]  # 只保留前N个检测结果

        except Exception as e:
            # 捕获推理过程中的错误并记录
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
        img = frame.copy()  # 复制输入帧
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
