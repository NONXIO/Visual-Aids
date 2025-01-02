import os

class TTSConfig:
    """TTS 模块配置"""

    # 默认语音名称（用于 pyttsx3）
    DEFAULT_VOICE_NAME = "Microsoft Huihui Desktop - Chinese (Simplified)"

    # 默认 pyttsx3 语速
    DEFAULT_RATE = 150
    # gTTS 播放速度（通过 ffplay 的 atempo 调整，范围 ~0.5~2.0）
    DEFAULT_PLAYBACK_SPEED = 1.5

    # 默认音量
    DEFAULT_VOLUME = 1.0

    # 默认语言（用于 gTTS）
    DEFAULT_LANGUAGE = "zh"

    # 是否优先使用在线 gTTS（网络不好自动切换到 pyttsx3）
    USE_ONLINE_TTS_FIRST = True

    # ffmpeg / ffplay 路径
    # 根据你实际位置设置
    FFMPEG_PATH = os.path.abspath(r"F:\Visual-Aids\models\tts_engine\ffmpeg.exe")
    FFPLAY_PATH = os.path.abspath(r"F:\Visual-Aids\models\tts_engine\ffplay.exe")

    # 临时音频文件存储路径
    TEMP_AUDIO_DIR = os.path.abspath(r"F:\Visual-Aids\logs\audio")
    if not os.path.exists(TEMP_AUDIO_DIR):
        os.makedirs(TEMP_AUDIO_DIR)
