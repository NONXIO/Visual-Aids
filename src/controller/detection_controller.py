# src/controller/detection_controller.py
import cv2
import time
from collections import deque
from src.detector.detection_utils import prioritize_detections, format_detection_speech
from src.detector.detection_config import DetectionConfig
from src.utils.logger import setup_logger


class DetectionController:
    """检测控制器，管理检测过程和TTS调用"""

    def __init__(self, detector, tts_engine):
        """
        初始化检测控制器

        Args:
            detector: 目标检测器
            tts_engine: TTS引擎
        """
        self.logger = setup_logger('DetectionController')
        self.detector = detector
        self.tts_engine = tts_engine
        self.detection_queue = deque(maxlen=DetectionConfig.QUEUE_MAX_SIZE)
        self.last_tts_data = {
            'time': 0,
            'content': ""
        }
        self.frame_counter = 0

    def process_frame(self, frame):
        """
        处理单个视频帧

        Args:
            frame: 输入视频帧

        Returns:
            numpy.ndarray: 处理后的帧，带有检测标记
        """
        self.frame_counter += 1
        process_this_frame = (self.frame_counter % DetectionConfig.PROCESS_EVERY_N_FRAMES == 0)

        if not process_this_frame:
            return frame

        try:
            # 执行目标检测
            detections = self.detector.detect(frame)

            # 确保检测结果是列表
            if not isinstance(detections, list):
                self.logger.warning(f"检测器返回了非列表类型的结果: {type(detections)}")
                detections = []

            # 优先级排序
            prioritized_detections = prioritize_detections(detections)

            # 更新检测队列
            current_time = time.time()
            self.detection_queue.append((current_time, prioritized_detections))

            # 移除过期检测结果
            while (self.detection_queue and
                   current_time - self.detection_queue[0][0] > DetectionConfig.DETECTION_HISTORY_SECONDS):
                self.detection_queue.popleft()

            # 处理TTS
            self._process_tts()

            # 绘制检测结果
            display_frame = self.detector.draw_detections(frame, prioritized_detections)
            return display_frame

        except Exception as e:
            self.logger.error(f"处理帧时发生错误: {str(e)}")
            return frame

    def _process_tts(self):
        """处理TTS语音播报"""
        if not self.detection_queue:
            return

        current_time = time.time()

        try:
            latest_detections = self.detection_queue[-1][1]  # 获取最新的检测结果

            # 检查是否有有效的检测结果
            if not isinstance(latest_detections, list):
                self.logger.warning(f"Invalid detection format: {type(latest_detections)}")
                return

            # 只有当有检测到物体时才进行处理
            if not latest_detections:
                return

            # 生成语音文本
            speech_text = format_detection_speech(latest_detections)

            # 只在内容变化或足够时间过去后播报
            if (speech_text and speech_text != self.last_tts_data['content'] or
                    current_time - self.last_tts_data['time'] >= DetectionConfig.TTS_THROTTLE_SECONDS):
                # 直接向TTS引擎传递检测结果列表，而不是文本
                self.tts_engine.speak(latest_detections)
                self.last_tts_data['content'] = speech_text
                self.last_tts_data['time'] = current_time
        except Exception as e:
            self.logger.error(f"处理TTS时发生错误: {str(e)}")