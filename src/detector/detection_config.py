class DetectionConfig:
    """检测系统配置参数"""

    # GUI配置
    USE_GUI = True  # 是否使用GUI界面（设置为False可在无GUI环境下运行）

    # 队列配置
    QUEUE_MAX_SIZE = 30
    DETECTION_HISTORY_SECONDS = 1.0

    # 性能优化
    PROCESS_EVERY_N_FRAMES = 2  # 每处理2帧中的1帧

    # TTS相关
    TTS_THROTTLE_SECONDS = 3.0  # TTS播报节流时间

    # 检测结果处理
    MAX_DETECTIONS = 5  # 最大检测结果数量

    # 视频设置
    VIDEO_PATH = "../data/test.mp4"
    WINDOW_NAME = "Visual Aids"

    # 对象优先级权重 (值越大优先级越高)
    OBJECT_PRIORITIES = {
        'person': 10,  # 行人优先级最高
        'traffic light': 9,  # 交通灯高优先级
        'stop sign': 9,  # 停止标志高优先级
        'car': 8,  # 汽车次优先级
        'bicycle': 7,  # 自行车
        'bus': 6,  # 公交车
        'motorbike': 6,  # 摩托车
        'truck': 5,  # 卡车
        'fire hydrant': 4,  # 消防栓
        # 可以根据需要添加更多类别
    }