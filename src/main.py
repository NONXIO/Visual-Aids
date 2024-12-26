from src.camera import Camera
from src.detector.yolo import ObjectDetector
from src.utils.logger import setup_logger
import cv2


def main():
    logger = setup_logger('main')
    camera = Camera()
    detector = ObjectDetector()

    try:
        logger.info("Starting camera...")
        camera.start()

        # 创建命名窗口
        window_name = "Visual Aids"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        logger.info("Starting detection loop...")
        while True:
            # 获取帧
            frame = camera.get_frame()

            # 如果成功获取到帧
            if frame is not None:
                # 运行目标检测
                detections = detector.detect(frame)

                # 绘制检测结果
                display_frame = detector.draw_detections(frame, detections)

                # 显示结果
                cv2.imshow(window_name, display_frame)

                # 输出检测到的物体（这里后续会替换为TTS）
                if detections:
                    objects = [f"{d['class']} ({d['confidence']:.2f})" for d in detections]
                    print(f"Detected: {', '.join(objects)}")

            # 检查键盘事件，等待1ms
            key = cv2.waitKey(1) & 0xFF

            # 如果按下ESC键 (27) 或点击窗口关闭按钮，退出循环
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Display window closed by user")
                break

    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt...")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        logger.info("Cleaning up...")
        camera.stop()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    main()