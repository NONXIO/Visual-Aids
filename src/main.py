import sys
import time
import cv2
import traceback
from src.camera.threaded_camera import ThreadedVideoCapture
from src.controller.detection_controller import DetectionController
from src.detector.detection_config import DetectionConfig
from src.utils.resource_manager import initialize_modules, cleanup_resources
from src.utils.logger import setup_logger


# 在main.py中改进主函数的退出处理

def main():
    """
    主程序入口，负责初始化模块并开始检测循环。
    """
    logger = setup_logger('main')
    cap = None
    tts_engine = None

    try:
        # 初始化各模块
        detector, tts_engine, ocr, _ = initialize_modules()

        # 创建检测控制器
        controller = DetectionController(detector, tts_engine)

        # 检查OpenCV是否支持GUI
        has_gui = True
        try:
            cv2.namedWindow(DetectionConfig.WINDOW_NAME, cv2.WINDOW_NORMAL)
            logger.info("成功创建显示窗口")
        except cv2.error as e:
            logger.warning(f"无法创建显示窗口，将在无GUI模式下运行: {str(e)}")
            has_gui = False

        # 启动线程化视频捕获
        try:
            cap = ThreadedVideoCapture(DetectionConfig.VIDEO_PATH)
            # 获取视频帧率
            fps = cap.fps
            frame_time = 1.0 / fps
            logger.info(f"视频帧率: {fps}，帧间隔: {frame_time:.4f}秒")
        except Exception as e:
            logger.error(f"初始化视频捕获失败: {str(e)}")
            raise

        logger.info("开始检测循环...")
        frame_count = 0
        last_process_time = time.time()

        while True:
            # 从视频流中读取帧
            ret, frame = cap.read()
            if not ret:
                logger.info("视频播放结束或读取帧失败")
                break

            current_time = time.time()
            frame_count += 1

            # 处理当前帧
            display_frame = controller.process_frame(frame)

            # 显示帧率信息
            if frame_count % 30 == 0:
                fps_actual = 30 / (current_time - last_process_time) if current_time != last_process_time else 0
                logger.info(f"当前处理帧率: {fps_actual:.2f} FPS")
                last_process_time = current_time

            # 显示处理后的帧（如果支持GUI）
            if has_gui:
                cv2.imshow(DetectionConfig.WINDOW_NAME, display_frame)
                # 检查键盘事件，按下 ESC 键或关闭窗口退出循环
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty(DetectionConfig.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("用户关闭窗口")
                    break
            else:
                # 在无GUI模式下，每100帧打印一次状态
                if frame_count % 100 == 0:
                    logger.info(f"已处理 {frame_count} 帧")

    except KeyboardInterrupt:
        logger.info("检测循环因键盘中断而停止...")
    except Exception as e:
        logger.error(f"检测循环中发生错误: {str(e)}\n{traceback.format_exc()}")
    finally:
        # 在finally块中包装cleanup_resources以确保无论如何都会执行，并且不会因异常而中断
        try:
            cleanup_resources(cap, tts_engine)
        except Exception as e:
            logger.error(f"清理资源时发生致命错误: {str(e)}")
        finally:
            logger.info("程序退出")
            sys.exit(0)

if __name__ == "__main__":
    main()