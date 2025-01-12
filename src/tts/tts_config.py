# src/tts/config.py
import os

class TTSConfig:
    """TTS 模块配置"""

    # 默认语音名称（用于 pyttsx3）
    DEFAULT_VOICE_NAME = "Microsoft Huihui Desktop - Chinese (Simplified)"

    # 默认 pyttsx3 语速
    DEFAULT_RATE = 175
    # gTTS 播放速度（通过 ffplay 的 atempo 调整，范围 ~0.5~2.0）
    DEFAULT_PLAYBACK_SPEED = 1.75

    # 默认音量
    DEFAULT_VOLUME = 1.0

    # 默认语言（用于 gTTS）
    DEFAULT_LANGUAGE = "zh"

    # 是否优先使用在线 gTTS（网络不好自动切换到 pyttsx3）
    USE_ONLINE_TTS_FIRST = True

    # ffmpeg / ffplay 路径
    # 自动获取当前所在的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 查找项目根目录
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    # 构造 tts_engine 目录的路径
    tts_engine_path = os.path.join(project_root, "models", "tts_engine")

    # 自动拼接完整路径
    FFMPEG_PATH = os.path.join(tts_engine_path, "ffmpeg.exe")
    FFPLAY_PATH = os.path.join(tts_engine_path, "ffplay.exe")

    # 临时音频文件存储路径
    TEMP_AUDIO_DIR = os.path.abspath(r"F:\Visual-Aids\logs\audio")
    if not os.path.exists(TEMP_AUDIO_DIR):
        os.makedirs(TEMP_AUDIO_DIR)

    # 最大语音播报距离（单位：米）
    MAX_SPEECH_DISTANCE = 3.0

    # 物体优先级分类（值越小优先级越高）
    OBJECT_PRIORITIES = {
        'person': 1,
        'bus': 2,
        'car': 3,
        'bicycle': 4,
        'motorbike': 5,
        'truck': 6,
        'traffic light': 7,
        'stop sign': 8,
        'fire hydrant': 9
    }
