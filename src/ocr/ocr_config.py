# src/ocr/config.py
class OCRConfig:
    """OCR 模块配置"""
    # 选择 OCR 模式：'paddle' 或 'tesseract'
    OCR_MODE = 'tesseract'  # 默认使用 Tesseract OCR
