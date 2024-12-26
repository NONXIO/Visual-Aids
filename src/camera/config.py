class CameraConfig:
    """相机模块配置"""

    # 设备配置
    DEVICE_ID = 0  # 相机设备ID

    # 分辨率配置
    FRAME_WIDTH = 1280  # 帧宽度
    FRAME_HEIGHT = 720  # 帧高度
    FPS = 30  # 帧率

    # 缓冲配置
    BUFFER_SIZE = 2  # 帧缓冲大小

    # 图像预处理配置
    BRIGHTNESS_ALPHA = 1.2  # 亮度调整系数
    BRIGHTNESS_BETA = 10  # 亮度调整偏移量