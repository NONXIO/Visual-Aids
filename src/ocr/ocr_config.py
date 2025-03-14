# src/ocr/config.py
class OCRConfig:
    """OCR 模块配置"""
    # PaddleOCR 配置可以在这里添加
    LANGUAGE = 'ch'  # 默认中文
    USE_GPU = True  # 默认不使用 GPU