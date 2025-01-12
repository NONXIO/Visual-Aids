import sys
import time
import cv2
import traceback
from collections import deque
from src.detector.yolo import ObjectDetector
from src.tts.TTSEngine import TTSEngine
from src.utils.logger import setup_logger
from src.ocr.ocr import OCR

def main():
    """
    主程序入口，负责初始化模块并开始检测循环。
    """
    logger = setup_logger('main')  # 设置日志记录器

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

    except Exception as e:
        logger.error(f"初始化模块时发生错误: {str(e)}\n{traceback.format_exc()}")
        return

    # 创建窗口以显示视频
    window_name = "Visual Aids"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    # 加载视频文件
    video_path = "../data/test.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return

    # 使用 deque 存储最近 1 秒内的检测结果
    detection_queue = deque(maxlen=30)

    try:
        logger.info("开始检测循环...")
        while True:
            # 从视频流中读取帧
            ret, frame = cap.read()
            if not ret:
                logger.info("视频播放结束或读取帧失败")
                break

            # 检测当前帧中的目标
            detections = detector.detect(frame)
            current_time = time.time()

            # 将检测结果加入队列，并移除超过 1 秒的旧结果
            detection_queue.append((current_time, detections))
            detection_queue = deque(
                [(t, d) for t, d in detection_queue if current_time - t <= 1], maxlen=30
            )

            # 如果有检测结果，进行语音播报
            if detection_queue:
                latest_detections = detection_queue[-1][1]  # 获取最新的检测结果
                speech_text = "检测到：" + "，".join(
                    f"{det['class']}，{det['distance']}米" for det in latest_detections
                )
                tts_engine.speak(speech_text)  # 播报检测到的目标

            # 绘制检测结果并显示帧
            display_frame = detector.draw_detections(frame, detections)
            cv2.imshow(window_name, display_frame)

            # 检查键盘事件，按下 ESC 键或关闭窗口退出循环
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("用户关闭窗口")
                break

    except KeyboardInterrupt:
        logger.info("检测循环因键盘中断而停止...")
    except Exception as e:
        logger.error(f"检测循环中发生错误: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.info("清理资源...")
        try:
            if 'tts_engine' in locals():
                tts_engine.stop()  # 停止 TTS 引擎

            cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
            cap.release()  # 释放视频捕获对象
            for _ in range(5):
                cv2.waitKey(1)  # 确保窗口完全关闭

            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"清理资源时发生错误: {str(e)}\n{traceback.format_exc()}")
        finally:
            sys.exit(0)

if __name__ == "__main__":
    main()

