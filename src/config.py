import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 各个目录的路径
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# 日志配置
LOG_CONFIG = {
    'filename': os.path.join(LOGS_DIR, 'app.log'),
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# 应用配置
APP_CONFIG = {
    'debug': True,  # 调试模式
    'language': 'zh_CN'  # 默认语言
}