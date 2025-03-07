# Visual-Aids: 为视障人士提供视觉辅助的系统

## **项目概述**
Visual-Aids 项目旨在帮助视障人士日常出行和活动。通过实时图像处理、目标检测、文本识别和语音合成，提供用户关于周围环境的语音指导。本系统模块化设计，具有良好的扩展性，并经过优化以便于在移动设备上部署。

### **核心功能**
- **实时目标检测**：识别用户路径中的障碍物、车辆和其他物体。
- **文本识别**：识别环境中的文本，例如公交车号、地铁标志等。
- **语音输出**：将检测到的物体或识别到的文本转化为语音，帮助用户理解周围环境。
- **路径导航（待实现）**：规划步行路径，用户偏离路径时提供提醒。
- **轻量化部署**：使用轻量化模型（如 YOLOv11s）和量化技术，优化移动设备的性能。

## **项目结构**

```
Visual-Aids
│
├── data/                  # 数据目录（如模型文件、临时数据）
├── docs/                  # 项目文档
├── logs/                  # 日志目录
│   ├── app.log            # 主程序日志
│   ├── audio              # 语音输出 mp3 文件
├── models/                # 模型文件目录
│   ├── tts_engine/        # 文字转语音引擎模型
│   │   ├── ffmpeg.exe     # FFmpeg 可执行文件
│   │   ├── ffplay.exe     # FFplay 可执行文件
│
├── src/                   # 核心源码
│   ├── camera/            # 相机捕获模块
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── config.py
│   │
│   ├── detector/          # 目标检测模块
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── yolo.py
│   │
│   ├── ocr/               # 文本识别模块（OCR）
│   │   ├── __init__.py
│   │   ├── ocr.py         # OCR 功能实现
│   │   ├── config.py      # OCR 配置文件
│   │
│   ├── tts/               # 文字转语音模块（TTS）
│   │   ├── __init__.py
│   │   ├── tts.py         # 语音合成模块
│   │   ├── TTSEngine.py   # 语音引擎接口
│   │   ├── config.py
│   │
│   ├── navigation/        # 导航与路径规划模块
│   │   ├── __init__.py
│   │   ├── navigation.py
│   │
│   ├── utils/             # 工具函数与配置模块
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py      # 日志记录
│   │
│   ├── main.py            # 主程序脚本
│
├── requirements.txt       # Python 依赖项
├── .gitignore             # Git 忽略规则
```

---

## **模块说明**

### **1. 相机模块**
- **用途**：捕获相机视频帧并进行预处理，以便后续处理。
- **关键文件**：
  - `camera.py`: 处理视频捕获、预处理和缓冲。
  - `config.py`: 包含相机相关配置（如分辨率、帧率）。
- **功能**：
  - 可调整分辨率和帧率。
  - 提供实时预处理以提高目标检测和文本识别的准确性。

---

### **2. 目标检测模块**
- **用途**：使用 YOLO 模型对视频帧中的目标进行检测。
- **关键文件**：
  - `yolo.py`: 基于 YOLOv11s 实现目标检测。
  - `config.py`: 定义检测配置（如模型路径、信度阈值）。
- **功能**：
  - 轻量化 YOLO 模型，支持实时检测。
  - 按目标类别过滤检测结果（如公交车、行人）。
  - 支持绘制边界框和估算障碍物距离。

---

### **3. 文本识别模块（OCR）**
- **用途**：从视频帧中识别文本信息，例如公交车号和地铁标志。
- **关键文件**：
  - `ocr.py`: 使用 Tesseract 或 PaddleOCR 实现 OCR 功能。默认使用 Tesseract OCR
  - `config.py`: 包含 OCR 模块的配置。
- **功能**：
  - 自动从检测到的区域提取文本。
  - 支持多语种文本识别（如中文、英文）。
  - 提供预处理功能以提升识别准确率。

#### **OCR 安装指南**
默认使用 Tesseract OCR
1. **Tesseract OCR**：
   - Windows 用户：从 [Tesseract GitHub 页面](https://github.com/UB-Mannheim/tesseract/wiki) 下载最新的安装包。
     - 安装路径建议使用默认值（如 `C:\Program Files\Tesseract-OCR`）。
     - 将安装路径添加到系统的 PATH 环境变量中。
   - Linux 用户：
     ```bash
     sudo apt update
     sudo apt install tesseract-ocr
     ```
   - Mac 用户：
     ```bash
     brew install tesseract
     ```
   - 安装完成后运行以下命令验证：
     ```bash
     tesseract --version
     ```

2. **PaddleOCR（可选）**：
   - 安装 PaddleOCR：
     ```bash
     pip install paddleocr
     ```
   - 如需 GPU 加速，请参考 [PaddleOCR 文档](https://github.com/PaddlePaddle/PaddleOCR)。

---

### **4. 文字转语音模块**
- **用途**：将文字转化为语音，提供语音指导。
- **关键文件**：
  - `tts.py`: 实现文字转语音功能，支持线上 gTTS 和离线 pyttsx3 自动切换。
  - `TTSEngine.py`: 优化 TTS 引擎，支持跨线程队列处理和带倾时应策。
  - `config.py`: 管理 TTS 配置，例如语言、语速和音量。
- **功能**：
  - 实时语音反馈，支持多线程播放。
  - 优先使用线上 gTTS，如有异常自动切换到 pyttsx3。
  - 支持多语种（如中文、英文）。
  - 对倾时程序进行自动处理，保证系统稳定。

---

### **5. 导航模块（待实现）**
- **用途**：规划步行路径，并在用户偏离路径时提供提醒。
- **关键文件**：
  - `navigation.py`: 使用 GPS 和地图 API 计算路径。
- **功能**：
  - 获取步行路径（例如通过 Google Maps 或 OpenStreetMap）。
  - 检测并提醒用户路径偏离情况。

---

### **6. 工具模块**
- **用途**：提供实用函数、日志记录和全局配置管理。
- **关键文件**：
  - `logger.py`: 设置和管理模块的日志记录。
  - `config.py`: 集中管理全局配置。
- **功能**：
  - 配置化日志记录。
  - 简化模块集成。

---

### **7. 主程序**
- **用途**：整合所有模块，实现系统的核心功能。
- **关键文件**：
  - `main.py`: 包含主程序循环，用于实时目标检测、文本识别和语音播报。
- **功能**：
  - 捕获相机视频帧。
  - 检测目标并提取文本信息。
  - 通过文字转语音模块提供语音反馈。

---

## **当前进度与未来计划**

### **已完成**
- 相机捕获和预处理。
- 使用 YOLO 进行目标检测。
- 集成文字转语音模块，提供实时语音反馈。
- 实现 OCR 功能，用于文本识别。

### **未来计划**
- 完善ocr逻辑以及ocr转语音逻辑
- 增加导航功能，提供路径偏离提醒。
- 优化模块以支持移动端部署（如 TensorFlow Lite 和 ONNX）。

---

## **如何运行项目**

1. **安装依赖项**：
   ```bash
   pip install -r requirements.txt
   ```

2. **准备环境**：
   - 确保 YOLO 模型文件位于 `models/` 目录中。
   - 安装必要的文字转语音和 OCR 依赖。

3. **运行程序**：
   ```bash
   python src/main.py
   ```

4. **调整配置**：
   - 在 `src/utils/config.py` 或模块专用的 `config.py` 文件中修改设置。

---

## **参与贡献**
欢迎贡献代码！请随时通过 Issue 或 Pull Request 提交功能改进建议或修复。

---

## **许可证**
本项目基于 MIT 许可证开源，详情请参阅 `LICENSE` 文件。
