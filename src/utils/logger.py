import logging
import os
from src.config import LOG_CONFIG


def setup_logger(name):
    """
    设置并返回一个命名的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(LOG_CONFIG['filename']), exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_CONFIG['level']))

    # 创建文件处理器
    file_handler = logging.FileHandler(LOG_CONFIG['filename'], encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger