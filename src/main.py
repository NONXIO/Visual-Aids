import sys
import cv2
import traceback
from src.camera.threaded_camera import ThreadedVideoCapture
from src.controller.detection_controller import DetectionController
from src.detector.detection_config import DetectionConfig
from src.utils.resource_manager import initialize_modules, cleanup_resources
from src.utils.logger import setup_logger


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
        except Exception as e:
            logger.error(f"初始化视频捕获失败: {str(e)}")
            raise

        logger.info("开始检测循环...")
        frame_count = 0

        while True:
            # 从视频流中读取帧
            ret, frame = cap.read()
            if not ret:
                logger.info("视频播放结束或读取帧失败")
                break

            frame_count += 1
            # 处理当前帧
            display_frame = controller.process_frame(frame)

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
                # 在无GUI模式下添加退出机制，例如处理特定数量的帧后退出
                # 这里可以根据需要调整或添加其他退出条件
                # if frame_count >= 1000:  # 例如处理1000帧后退出
                #     logger.info("已达到预设帧数限制，退出程序")
                #     break

    except KeyboardInterrupt:
        logger.info("检测循环因键盘中断而停止...")
    except Exception as e:
        logger.error(f"检测循环中发生错误: {str(e)}\n{traceback.format_exc()}")
    finally:
        cleanup_resources(cap, tts_engine)
        sys.exit(0)


if __name__ == "__main__":
    main()