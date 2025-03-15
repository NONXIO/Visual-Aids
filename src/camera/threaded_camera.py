# src/camera/threaded_camera.py
import cv2
import threading
import time
from src.utils.logger import setup_logger


class ThreadedVideoCapture:
    """线程化视频捕获类，提高视频读取性能"""

    def __init__(self, source):
        """
        初始化线程化视频捕获

        Args:
            source: 视频源路径或摄像头ID
        """
        self.logger = setup_logger('ThreadedVideoCapture')
        self.logger.info(f"初始化视频捕获: {source}")
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            self.logger.error(f"无法打开视频源: {source}")
            raise ValueError(f"无法打开视频源: {source}")

        # 获取视频的原始帧率
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.logger.info(f"视频原始帧率: {self.original_fps}")

        # 如果是文件视频且帧率有效，则使用它，否则默认使用30fps
        self.fps = self.original_fps if self.original_fps > 0 else 30.0
        self.frame_time = 1.0 / self.fps

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()

        # 添加帧计时器，用于控制帧率
        self.last_frame_time = time.time()

        # 创建并启动线程
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        self.logger.info("视频捕获线程已启动")

    def _update(self):
        """后台线程：持续读取视频帧，但尊重原始帧率"""
        while self.running:
            if not self.cap.isOpened():
                self.logger.error("视频源已关闭")
                self.running = False
                break

            # 添加帧率控制
            current_time = time.time()
            elapsed = current_time - self.last_frame_time

            # 等待直到达到正确的帧间隔
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 读取下一帧
            ret, frame = self.cap.read()
            self.last_frame_time = time.time()

            with self.lock:
                self.ret, self.frame = ret, frame

            if not ret:
                self.logger.info("视频播放结束")
                self.running = False
                break

    def read(self):
        """
        读取当前帧

        Returns:
            tuple: (ret, frame) 其中ret表示是否读取成功，frame是图像帧
        """
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else None

    def release(self):
        """释放视频资源"""
        self.logger.info("正在释放视频资源...")
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        self.logger.info("视频资源已释放")