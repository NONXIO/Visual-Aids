import sys
import time
import cv2
import traceback
from src.detector.yolo import ObjectDetector
from src.tts.TTSEngine import TTSEngine
from src.utils.logger import setup_logger
from src.ocr.ocr import OCR

def main():
    # 初始化日志
    logger = setup_logger('main')

    # 初始化模块
    try:
        logger.info("初始化目标检测器...")
        detector = ObjectDetector()
        logger.info("目标检测器初始化成功")

        logger.info("初始化 TTS 引擎...")
        tts_engine = TTSEngine()
        logger.info("TTS 引擎初始化成功")

        logger.info("初始化 OCR 模块...")
        ocr = OCR()
        logger.info("OCR 模块初始化成功")

    except Exception as e:
        logger.error(f"初始化模块时发生错误: {str(e)}\n{traceback.format_exc()}")
        return

    # 创建窗口
    window_name = "Visual Aids"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    # 使用视频文件代替相机输入
    video_path = "../data/test.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return

    last_speech_time = 0  # 上次语音播报的时间
    speech_interval = 3  # 设置语音播报间隔为3秒

    try:
        logger.info("开始检测循环...")
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.info("视频播放结束或读取帧失败")
                break

            # 运行目标检测
            detections = detector.detect(frame)

            # 获取当前时间
            current_time = time.time()

            # 限制语音播报频率
            if current_time - last_speech_time >= speech_interval:
                last_speech_time = current_time  # 更新上次语音播报时间

                if detections:
                    descriptions = []
                    for detection in detections:
                        cls = detection['class']
                        distance = detection['distance']
                        descriptions.append(f"{cls}，{distance}米")

                    # 拼接检测结果
                    speech_text = "检测到：" + "，".join(descriptions)

                    # 播报语音
                    tts_engine.speak(speech_text)

            # 绘制检测结果
            display_frame = detector.draw_detections(frame, detections)

            # 显示处理后的帧
            cv2.imshow(window_name, display_frame)

            # 检查键盘事件，等待 1 毫秒
            key = cv2.waitKey(1) & 0xFF

            # 按下 ESC 键 (27) 或关闭窗口退出循环
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
            # 先停止 TTS 引擎
            if 'tts_engine' in locals():
                tts_engine.stop()

            # 关闭窗口和释放视频资源
            cv2.destroyAllWindows()
            cap.release()

            # 确保窗口真正关闭
            for _ in range(5):
                cv2.waitKey(1)

            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"清理资源时发生错误: {str(e)}\n{traceback.format_exc()}")
        finally:
            sys.exit(0)

if __name__ == "__main__":
    main()
