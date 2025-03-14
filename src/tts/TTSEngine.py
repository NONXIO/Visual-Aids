# src/tts/TTSEngine.py
import threading
import queue
from .tts import TextToSpeech
from src.utils.logger import setup_logger
from src.tts.tts_config import TTSConfig

class TTSEngine:
    """
    优化后的 TTS 引擎：
      - 通过一个工作线程从队列中取要播报的文本；
      - 在 stop() 中退出，不再强行杀线程；
      - 使用在线 gTTS 进行文本到语音的转换。
    """

    def __init__(self):
        self.logger = setup_logger("TTSEngine")

        # 初始化 TTS 播报器
        self.tts = TextToSpeech()
        # 存放待播报文本的队列
        self.queue = queue.Queue()
        # 当需要停止时，触发此事件，让子线程优雅退出
        self.stop_event = threading.Event()
        # 标记当前是否正在"说话"（避免 stop() 过程中出现竞争）
        self.is_speaking = threading.Event()

        # 最大播报距离
        self.max_speech_distance = TTSConfig.MAX_SPEECH_DISTANCE
        # 物体优先级字典
        self.object_priorities = TTSConfig.OBJECT_PRIORITIES

        # 创建工作线程（非 daemon），方便在主线程中 join 等待其退出
        self.worker_thread = threading.Thread(
            target=self._process_queue,
            name="TTSWorkerThread",
            daemon=False
        )
        self.worker_thread.start()

    def speak(self, text):
        """
        将文本放入队列中等待播报。

        Args:
            text (str): 要播报的文本。
        """
        if not self.stop_event.is_set():
            self.queue.put(text)

    def _process_queue(self):
        """
        子线程函数：持续从队列中获取需要播报的文本，逐条调用 self.tts.speak().
        如果 stop_event 被设置，就退出循环。
        """
        self.logger.info("TTS 工作线程启动。")
        while not self.stop_event.is_set():
            try:
                # 设一个小超时，避免一直阻塞
                text = self.queue.get(timeout=0.1)
            except queue.Empty:
                # 队列空，继续循环检查 stop_event
                continue

            if text is not None:
                self.is_speaking.set()
                try:
                    self.tts.speak(text)
                except Exception as e:
                    self.logger.error(f"处理文本 '{text}' 时发生错误: {str(e)}")
                finally:
                    self.is_speaking.clear()
                self.queue.task_done()

        self.logger.info("TTS 工作线程检测到停止事件，已退出。")

    def stop(self):
        """
        停止 TTS 引擎：
          - 通知子线程停止处理队列；
          - 清空队列；
          - 停止正在进行的音频播放；
          - join 子线程，等待退出。
        """
        self.logger.info("正在停止 TTS 引擎...")
        # 通知子线程退出循环
        self.stop_event.set()

        # 清空队列，防止还有大量未处理的文本
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

        # 如果此时正在说话，先尝试停止
        if self.is_speaking.is_set():
            self.tts.stop()

        # 等待子线程退出
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                self.logger.warning("TTS 工作线程未能在超时时间内正常退出，可能还在阻塞。")
            else:
                self.logger.info("TTS 工作线程已退出。")

        self.logger.info("TTS 引擎已停止。")