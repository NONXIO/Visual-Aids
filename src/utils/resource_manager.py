# src/utils/resource_manager.py
import cv2
import traceback
from src.detector.yolo import ObjectDetector
from src.tts.TTSEngine import TTSEngine
from src.ocr.ocr import OCR
from src.utils.logger import setup_logger


def initialize_modules():
    """
    初始化检测器、TTS和OCR模块

    Returns:
        tuple: (detector, tts_engine, ocr, logger)

    Raises:
        Exception: 初始化失败时抛出
    """
    logger = setup_logger('init')

    try:
        # 初始化目标检测模块
        logger.info("初始化目标检测器...")
        detector = ObjectDetector()
        logger.info("目标检测器初始化成功")

        # 初始化 TTS 引擎
        logger.info("初始化 TTS 引擎...")
        tts_engine = TTSEngine()
        logger.info("TTS 引擎初始化成功")

        # 初始化 OCR 模块
        logger.info("初始化 OCR 模块...")
        ocr = OCR()
        logger.info("OCR 模块初始化成功")

        return detector, tts_engine, ocr, logger
    except Exception as e:
        logger.error(f"初始化模块时发生错误: {str(e)}\n{traceback.format_exc()}")
        raise


def cleanup_resources(cap=None, tts_engine=None):
    """
    清理所有资源

    Args:
        cap: 视频捕获对象
        tts_engine: TTS引擎对象
    """
    logger = setup_logger('cleanup')
    logger.info("清理资源...")

    # 首先停止TTS引擎
    if tts_engine is not None:
        try:
            tts_engine.stop()
            logger.info("TTS引擎已停止")
        except Exception as e:
            logger.error(f"停止TTS引擎时发生错误: {str(e)}")

    # 释放视频捕获
    if cap is not None:
        try:
            cap.release()
            logger.info("视频资源已释放")
        except Exception as e:
            logger.error(f"释放视频资源时发生错误: {str(e)}")

    # 关闭OpenCV窗口
    try:
        cv2.destroyAllWindows()
        # 确保窗口正确关闭
        for _ in range(5):
            cv2.waitKey(1)
        logger.info("OpenCV窗口已关闭")
    except Exception as e:
        logger.error(f"关闭OpenCV窗口时发生错误: {str(e)}")

    logger.info("资源清理完成")