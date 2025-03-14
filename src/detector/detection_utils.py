# src/detector/detection_utils.py
from src.detector.detection_config import DetectionConfig
from src.detector.yolo_config import DetectorConfig  # 添加这个导入
from src.utils.logger import setup_logger

logger = setup_logger('detection_utils')


def prioritize_detections(detections, max_items=None):
    """
    根据类别和距离对检测结果进行优先级排序

    Args:
        detections (List[Dict]): 检测结果列表
        max_items (int, optional): 返回的最大结果数量，默认使用配置中的值

    Returns:
        List[Dict]: 优先级排序后的检测结果列表
    """
    if max_items is None:
        max_items = DetectionConfig.MAX_DETECTIONS

    # 确保输入是列表
    if not isinstance(detections, list):
        logger.warning(f"prioritize_detections收到非列表输入: {type(detections)}")
        return []

    # 检查是否有检测结果
    if not detections:
        return []

    try:
        # 为每个检测结果打分
        scored_detections = []
        for det in detections:
            # 确保检测结果是字典且包含所需键
            if not isinstance(det, dict) or 'class' not in det or 'distance' not in det:
                logger.warning(f"跳过无效的检测结果: {det}")
                continue

            # 基于类别优先级的分数
            # 使用detection_config.py中的OBJECT_PRIORITIES（如果存在）
            if hasattr(DetectionConfig, 'OBJECT_PRIORITIES'):
                class_score = DetectionConfig.OBJECT_PRIORITIES.get(det['class'], 1)
            else:
                # 否则使用yolo_config.py中的OBJECT_REAL_WIDTHS作为后备
                class_score = DetectorConfig.OBJECT_REAL_WIDTHS.get(det['class'], 0.5) * 2

            # 距离越近分数越高
            distance_score = 10 / (det['distance'] + 1)
            # 组合分数
            total_score = class_score * distance_score
            scored_detections.append((total_score, det))

        # 按分数排序并取前N个
        # 修复比较问题：使用第一个元素(score)作为比较键
        scored_detections.sort(key=lambda x: x[0], reverse=True)  # 按分数从高到低排序
        return [det for _, det in scored_detections][:max_items]
    except Exception as e:
        logger.error(f"优先级排序时发生错误: {str(e)}")
        return []


def format_detection_speech(detections):
    """
    将检测结果格式化为语音文本

    Args:
        detections (List[Dict]): 检测结果列表

    Returns:
        str: 格式化后的语音文本
    """
    if not detections:
        return ""

    try:
        descriptions = []
        for det in detections:
            if not isinstance(det, dict):
                logger.warning(f"跳过非字典类型的检测结果: {type(det)}")
                continue

            if 'class' not in det or 'distance' not in det:
                logger.warning(f"检测结果缺少必要字段: {det}")
                continue

            descriptions.append(f"{det['class']}，{det['distance']}米")

        if not descriptions:
            return ""

        return "检测到：" + "，".join(descriptions)
    except Exception as e:
        logger.error(f"格式化检测结果时发生错误: {str(e)}")
        return "检测到物体"