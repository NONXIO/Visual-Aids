from tts.tts import TextToSpeech
from src.tts.config import TTSConfig

# 创建 TTS 实例
tts = TextToSpeech()

# 测试 gTTS 播报
print("测试 gTTS 播报:")
tts.speak("检测到前方0.5米有人", speed=1.5)

# 测试 pyttsx3 播报
print("测试 pyttsx3 播报:")
TTSConfig.USE_ONLINE_TTS_FIRST = False
tts.speak("检测到前方0.5米有人")
