# src/tts/tts.py
import os
import tempfile
import subprocess
import pyttsx3
import threading
import time
from gtts import gTTS
from src.tts.tts_config import TTSConfig
from src.utils.logger import setup_logger


class TextToSpeech:
    """
    TextToSpeech 类：负责文本转语音的功能。

    特性：
    - 优先使用在线 gTTS，若失败则回退到离线 pyttsx3。
    - 支持动态中断当前语音播报，确保最新检测内容及时播报。
    """

    def __init__(self):
        self.logger = setup_logger("TTS")  # 初始化日志记录器

        # 初始化语言和优先使用在线TTS的配置
        self.language = TTSConfig.DEFAULT_LANGUAGE
        self.use_online_tts_first = TTSConfig.USE_ONLINE_TTS_FIRST

        # 初始化 pyttsx3 引擎和相关锁
        self.engine_lock = threading.Lock()
        self._initialize_pyttsx3()

        # ffplay 播放器相关锁和进程
        self.process_lock = threading.Lock()
        self.current_process = None

        # 播报状态锁
        self.speaking_lock = threading.Lock()
        self.speaking = False

    def _initialize_pyttsx3(self):
        """
        初始化 pyttsx3 引擎并设置默认语音属性。
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', TTSConfig.DEFAULT_RATE)  # 设置语速
            self.engine.setProperty('volume', TTSConfig.DEFAULT_VOLUME)  # 设置音量

            # 尝试设置指定的语音
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if TTSConfig.DEFAULT_VOICE_NAME in voice.name:
                    self.engine.setProperty('voice', voice.id)
                    break
            else:
                self.logger.warning(f"未找到指定语音 '{TTSConfig.DEFAULT_VOICE_NAME}'，使用默认语音。")
        except Exception as e:
            self.logger.error(f"[TTS 错误] 初始化 pyttsx3 失败: {str(e)}")
            self.engine = None

    def speak(self, text: str, speed: float = TTSConfig.DEFAULT_PLAYBACK_SPEED):
        """
        进行语音播报。

        Args:
            text (str): 要播报的文本内容。
            speed (float): 播报速度（在线模式下生效）。
        """
        self.interrupt()  # 中断当前语音播报
        with self.speaking_lock:
            self.speaking = True

        try:
            if self.use_online_tts_first:
                online_success = self._speak_online(text, speed)
                if not online_success:
                    self._speak_offline(text)
            else:
                self._speak_offline(text)
        finally:
            with self.speaking_lock:
                self.speaking = False

    def interrupt(self):
        """
        中断当前语音播报。
        """
        self.stop()  # 停止当前播放

    def _speak_online(self, text: str, speed: float) -> bool:
        """
        使用 gTTS 在线生成语音并通过 ffplay 播放。

        Args:
            text (str): 要播报的文本内容。
            speed (float): 播放速度。

        Returns:
            bool: 在线播报是否成功。
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=TTSConfig.TEMP_AUDIO_DIR) as temp_file:
                temp_path = temp_file.name

            # 生成语音文件
            tts = gTTS(text=text, lang=self.language)
            tts.save(temp_path)
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"音频文件生成失败: {temp_path}")

            # 播放生成的音频
            self._play_audio_with_ffplay(temp_path, speed)
            os.remove(temp_path)
            return True
        except Exception as e:
            self.logger.error(f"[TTS 错误] gTTS + ffplay 播放失败，将尝试离线 TTS: {str(e)}")
            return False

    def _play_audio_with_ffplay(self, audio_path: str, speed: float):
        """
        使用 ffplay 播放音频。

        Args:
            audio_path (str): 音频文件路径。
            speed (float): 播放速度。
        """
        with self.process_lock:
            try:
                self.current_process = subprocess.Popen(
                    [
                        TTSConfig.FFPLAY_PATH,
                        "-nodisp",
                        "-autoexit",
                        "-af", f"atempo={speed}",
                        audio_path
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False
                )
            except Exception as e:
                self.current_process = None
                raise RuntimeError(f"启动 ffplay 进程失败: {e}")

        # 等待播放结束或超时
        start_time = time.time()
        while True:
            if self.current_process.poll() is not None:
                break
            if time.time() - start_time > 15:
                self.logger.warning("ffplay 播放超时，强制终止进程")
                self._terminate_ffplay()
                break
            time.sleep(0.1)

        with self.process_lock:
            self.current_process = None

    def _terminate_ffplay(self):
        """
        强制终止 ffplay 进程。
        """
        with self.process_lock:
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=1.0)
                except Exception as e:
                    self.logger.error(f"终止 ffplay 进程失败: {e}")
                    self.current_process.kill()

    def _speak_offline(self, text: str):
        """
        使用 pyttsx3 进行离线语音播报。

        Args:
            text (str): 要播报的文本内容。
        """
        if not self.engine:
            self.logger.error("pyttsx3 引擎初始化失败，无法离线播报！")
            return

        with self.engine_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                self.logger.error(f"[TTS 错误] pyttsx3 播报失败: {str(e)}")

    def stop(self):
        """
        停止当前所有语音播报，包括 ffplay 和 pyttsx3。
        """
        self._terminate_ffplay()
        with self.engine_lock:
            if self.speaking:
                try:
                    self.engine.stop()
                except Exception as e:
                    self.logger.error(f"停止 pyttsx3 引擎失败: {str(e)}")
                finally:
                    self.speaking = False
