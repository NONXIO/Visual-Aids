# src/tts/tts.py
import os
import tempfile
import subprocess
import pyttsx3
import threading
import time
from gtts import gTTS
from src.tts.config import TTSConfig
from src.utils.logger import setup_logger



class TextToSpeech:
    """
    优化后的 TextToSpeech 类：
      - 当网络正常时，先使用 gTTS + ffplay 播报；
      - 如果播放失败，则自动 fallback 到 pyttsx3；
      - stop() 时可终止 ffplay 子进程和 pyttsx3.runAndWait()。
    """

    def __init__(self):
        self.logger = setup_logger("TTS")

        self.language = TTSConfig.DEFAULT_LANGUAGE
        # 是否优先使用在线 TTS（可在 config 里配置，也可在运行时修改）
        self.use_online_tts_first = TTSConfig.USE_ONLINE_TTS_FIRST

        # 初始化 pyttsx3 引擎
        self.engine_lock = threading.Lock()
        self._initialize_pyttsx3()

        # 用于在线播放时的 ffplay 进程
        self.process_lock = threading.Lock()
        self.current_process = None

        self.speaking_lock = threading.Lock()
        self.speaking = False

    def _initialize_pyttsx3(self):
        """
        初始化 pyttsx3 引擎并设置默认属性
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', TTSConfig.DEFAULT_RATE)
            self.engine.setProperty('volume', TTSConfig.DEFAULT_VOLUME)

            voices = self.engine.getProperty('voices')
            for voice in voices:
                if TTSConfig.DEFAULT_VOICE_NAME in voice.name:
                    self.engine.setProperty('voice', voice.id)
                    self.logger.debug(f"已设置 pyttsx3 语音: {voice.name}")
                    break
            else:
                self.logger.warning(f"未找到指定的语音 '{TTSConfig.DEFAULT_VOICE_NAME}'，使用默认: {voices[0].name}")

        except Exception as e:
            self.logger.error(f"[TTS 错误] 初始化 pyttsx3 失败: {str(e)}")
            self.engine = None

    def speak(self, text: str, speed: float = TTSConfig.DEFAULT_PLAYBACK_SPEED):
        """
        语音播报：
         - 默认先尝试在线 TTS (gTTS + ffplay)，若失败则使用离线 pyttsx3。
         - 如果 TTSConfig.USE_ONLINE_TTS_FIRST = False，则直接用离线 pyttsx3。
        """
        with self.speaking_lock:
            self.speaking = True

        try:
            if self.use_online_tts_first:
                # 如果在线失败或抛异常，会自动转到离线播报
                online_success = self._speak_online(text, speed)
                if not online_success:
                    self._speak_offline(text)
            else:
                # 直接用离线
                self._speak_offline(text)
        finally:
            with self.speaking_lock:
                self.speaking = False

    def _speak_online(self, text: str, speed: float) -> bool:
        """
        在线 TTS：使用 gTTS 生成临时 MP3，再用 ffplay 播放。
        返回 True / False 表示在线播放是否成功。
        """
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp3", dir=TTSConfig.TEMP_AUDIO_DIR
            ) as temp_file:
                temp_path = temp_file.name

            # 生成语音文件
            tts = gTTS(text=text, lang=self.language)
            tts.save(temp_path)
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"音频文件生成失败: {temp_path}")

            # 播放 mp3
            self._play_audio_with_ffplay(temp_path, speed)
            # 播放结束后删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return True

        except Exception as e:
            self.logger.error(f"[TTS 错误] gTTS + ffplay 播放失败，将尝试离线 TTS: {str(e)}")
            return False

    def _play_audio_with_ffplay(self, audio_path: str, speed: float):
        """
        用 ffplay 播放音频，带超时和 stop_event 检测。
        如果需要强制停止，可在 stop() 中终止 ffplay 进程。
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
                # 播放已结束
                break
            if time.time() - start_time > 15:  # 你也可调大或写到 config
                self.logger.warning("ffplay 播放超时，强制终止进程")
                self._terminate_ffplay()
                break
            time.sleep(0.1)

        with self.process_lock:
            self.current_process = None

    def _terminate_ffplay(self):
        """
        强制终止 ffplay 进程
        """
        with self.process_lock:
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=1.0)
                except Exception as e:
                    self.logger.error(f"终止 ffplay 进程失败: {e}")
                    self.current_process.kill()
                finally:
                    self.logger.debug("ffplay 进程已被终止")

    def _speak_offline(self, text: str):
        """
        离线 TTS：使用 pyttsx3
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
        停止正在播放的音频（ffplay）和 pyttsx3。
        在 TTSEngine.stop() 中会调用此方法，用于优雅退出。
        """
        # 先停止 ffplay
        self._terminate_ffplay()

        # 再停止 pyttsx3
        with self.speaking_lock:
            if self.speaking:
                with self.engine_lock:
                    try:
                        self.engine.stop()
                    except Exception as e:
                        self.logger.error(f"停止 pyttsx3 引擎失败: {str(e)}")
                    finally:
                        self.speaking = False
