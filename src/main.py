import sys
import time
import cv2
import traceback

from src.camera import Camera
from src.detector.yolo import ObjectDetector
from src.tts.TTSEngine import TTSEngine
from src.utils.logger import setup_logger


def check_camera_availability():
    """检查相机是否可用"""
    while True:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
                return
            else:
                raise Exception("相机被占用")
        except Exception:
            print("[警告] 相机被占用，等待释放...")
            time.sleep(1)  # 等待 1 秒后重试


def main():
    # 初始化日志
    logger = setup_logger('main')

    # 检查相机可用性
    logger.info("检查相机是否可用...")
    check_camera_availability()

    # 初始化模块
    try:
        logger.info("初始化相机...")
        camera = Camera()
        camera.start()
        logger.info("相机初始化成功")

        logger.info("初始化目标检测器...")
        detector = ObjectDetector()
        logger.info("目标检测器初始化成功")

        logger.info("初始化 TTS 引擎...")
        tts_engine = TTSEngine()
        logger.info("TTS 引擎初始化成功")

    except Exception as e:
        logger.error(f"初始化模块时发生错误: {str(e)}\n{traceback.format_exc()}")
        return

    # 创建窗口
    window_name = "Visual Aids"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    try:
        logger.info("开始检测循环...")
        while True:
            # 从相机获取帧
            frame = camera.get_frame()

            # 如果成功获取帧
            if frame is not None:
                # 运行目标检测
                detections = detector.detect(frame)

                # 如果有检测结果
                if detections:
                    descriptions = []
                    for detection in detections:
                        cls = detection['class']
                        confidence = detection['confidence']
                        distance = detection['distance']
                        descriptions.append(f"{cls}，距离 {distance} 米")
                        print(f"检测到 {cls}，置信度：{confidence:.2f}，距离：{distance} 米")

                    # 拼接成一句话，加入 TTS 队列
                    speech_text = "检测到：" + "，".join(descriptions)
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

            # 关闭窗口和停止相机
            cv2.destroyAllWindows()
            if 'camera' in locals():
                camera.stop()

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
