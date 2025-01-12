# src/camera/camera.py
import cv2
import threading
from queue import Queue
from .camera_config import CameraConfig
from src.utils.logger import setup_logger

class Camera:
    """相机类，负责视频捕获和预处理"""

    def __init__(self):
        """初始化相机"""
        self.config = CameraConfig()
        self.logger = setup_logger('camera')
        self.frame_queue = Queue(maxsize=self.config.BUFFER_SIZE)
        self.running = False
        self.cap = None
        self.capture_thread = None

    def _initialize_device(self):
        """初始化相机设备"""
        try:
            self.cap = cv2.VideoCapture(self.config.DEVICE_ID)

            # 设置相机参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)

            if not self.cap.isOpened():
                raise RuntimeError("无法打开相机设备")

            self.logger.info("相机设备初始化成功")

        except Exception as e:
            self.logger.error(f"相机设备初始化失败: {str(e)}")
            raise

    def _preprocess_frame(self, frame):
        """
        预处理捕获的帧

        Args:
            frame: 原始帧

        Returns:
            处理后的帧
        """
        try:
            # 调整亮度和对比度以改善文字识别
            processed = cv2.convertScaleAbs(
                frame,
                alpha=self.config.BRIGHTNESS_ALPHA,
                beta=self.config.BRIGHTNESS_BETA
            )
            return processed
        except Exception as e:
            self.logger.error(f"帧预处理失败: {str(e)}")
            return frame

    def _capture_loop(self):
        """相机捕获循环"""
        self.logger.info("开始相机捕获循环")
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # 预处理帧
                    processed_frame = self._preprocess_frame(frame)

                    # 如果队列已满，移除最旧的帧
                    if self.frame_queue.full():
                        self.frame_queue.get()

                    self.frame_queue.put(processed_frame)
                else:
                    self.logger.warning("帧捕获失败")

            except Exception as e:
                self.logger.error(f"捕获循环出错: {str(e)}")

    def start(self):
        """启动相机"""
        try:
            self._initialize_device()
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.start()
            self.logger.info("相机启动成功")
        except Exception as e:
            self.logger.error(f"相机启动失败: {str(e)}")
            self.stop()
            raise

    def stop(self):
        """停止相机"""
        self.logger.info("正在停止相机...")
        self.running = False

        # 等待捕获线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join()

        # 释放相机资源
        if self.cap:
            self.cap.release()

        # 清空帧队列
        while not self.frame_queue.empty():
            self.frame_queue.get()

        self.logger.info("相机已停止")

    def get_frame(self):
        """
        获取最新的帧

        Returns:
            最新的帧，如果没有可用的帧则返回None
        """
        try:
            return None if self.frame_queue.empty() else self.frame_queue.get()
        except Exception as e:
            self.logger.error(f"获取帧失败: {str(e)}")
            return None