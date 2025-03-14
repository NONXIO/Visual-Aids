# src/ocr/ocr.py
import cv2
from paddleocr import PaddleOCR
from src.utils.logger import setup_logger


class OCR:
    """OCR 模块，用于从图像中提取文本信息，使用在线 PaddleOCR"""

    def __init__(self):
        """
        初始化 OCR 模块，使用 PaddleOCR
        """
        self.logger = setup_logger('ocr')
        self._initialize_ocr_engine()  # 初始化 OCR 引擎

    def _initialize_ocr_engine(self):
        """初始化 PaddleOCR 引擎"""
        self.logger.info("初始化 PaddleOCR 模式")
        self.ocr_engine = PaddleOCR(use_gpu=False)

    def preprocess_image(self, image):
        """
        对输入图像进行预处理以提高 OCR 准确率。

        Args:
            image (numpy.ndarray): 输入图像。

        Returns:
            numpy.ndarray: 预处理后的图像。
        """
        try:
            # 转为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 二值化处理
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 去噪（可选）
            denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)
            return denoised
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            return image  # 返回原始图像以防止中断

    def clean_text(self, text):
        """
        清理 OCR 提取的文本，去除无效字符和空行。

        Args:
            text (str): OCR 提取的原始文本。

        Returns:
            str: 清理后的文本。
        """
        cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
        return " ".join(cleaned_lines)

    def extract_text(self, image):
        """
        从图像中提取文本

        Args:
            image (numpy.ndarray): 输入图像

        Returns:
            str: 提取的文本
        """
        try:
            preprocessed_image = self.preprocess_image(image)  # 图像预处理

            # 使用 PaddleOCR
            results = self.ocr_engine.ocr(preprocessed_image, cls=True)
            text = "\n".join([line[1][0] for line in results[0]]) if results else ""

            # 清理并格式化提取的文本
            cleaned_text = self.clean_text(text)
            self.logger.info(f"OCR 提取文本: {cleaned_text}")
            return cleaned_text

        except Exception as e:
            self.logger.error(f"OCR 提取失败: {str(e)}")
            return ""