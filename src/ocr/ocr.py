# src/ocr/ocr.py
import cv2
import pytesseract
import tempfile
import os
from paddleocr import PaddleOCR
from src.ocr.config import OCRConfig
from src.utils.logger import setup_logger


class OCR:
    def __init__(self):
        """
        初始化 OCR 模块，根据配置选择使用 PaddleOCR 或 Tesseract OCR
        """
        self.logger = setup_logger('ocr')

        if OCRConfig.OCR_MODE == 'paddle':
            self.logger.info("初始化 PaddleOCR 模式")
            self.ocr_engine = PaddleOCR(use_gpu=False)
            self.use_paddle_ocr = True
        elif OCRConfig.OCR_MODE == 'tesseract':
            self.logger.info("初始化 Tesseract OCR 模式")
            pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            self.use_paddle_ocr = False
        else:
            raise ValueError(f"未知的 OCR 模式: {OCRConfig.OCR_MODE}")

    def extract_text(self, image):
        """
        从图像中提取文本

        Args:
            image (numpy.ndarray): 输入图像

        Returns:
            str: 提取的文本
        """
        try:
            if self.use_paddle_ocr:
                # 使用 PaddleOCR
                results = self.ocr_engine.ocr(image, cls=True)
                if not results or len(results[0]) == 0:
                    self.logger.warning("PaddleOCR 未检测到文本")
                    return ""
                text = "\n".join([line[1][0] for line in results[0]])
            else:
                # 使用 Tesseract OCR
                temp_file_path = None
                try:
                    # 创建唯一临时文件
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        cv2.imwrite(temp_file_path, image)

                    # 调用 Tesseract OCR
                    text = pytesseract.image_to_string(temp_file_path, lang='chi_sim+eng')
                finally:
                    # 删除临时文件
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception as e:
                            self.logger.warning(f"无法删除临时文件: {temp_file_path}, 错误: {str(e)}")

            self.logger.info(f"OCR 提取文本: {text}")
            return text.strip()
        except Exception as e:
            self.logger.error(f"OCR 提取失败: {str(e)}")
            return ""
