# 🌉 BridgeVision Detection API
**BridgeVision** is a real-time API for detecting bridges in videos and images. Built in Python, the project provides flexible integration with external video processing systems.

## 🚀 Features
- **Dual Operation Modes**:
  Independent **REST** and **UDP** servers — choose the one that best suits your latency and scalability requirements.
- **High Performance**:
  **UDP** transport support enables low-latency frame processing.
- **Graceful Shutdown**:
  The server shuts down cleanly via `Ctrl+C` or `SIGTERM` — ideal for Docker and orchestration tools.
- **Ready to Use**:
  Distributed as a `.whl` package — install with a single command.

## 🧪 Technologies
- **Python 3.8+**
- **Flask** — web server
- **Ultralytics YOLO** — object detection
- **OpenCV, Pillow, NumPy** — image processing
- **UDP/JSON/base64** — efficient data exchange

## 📦 Установка
```bash
pip install bridges_detection_api-X.X.X-py3-none-any.whl
```
Replace `X.X.X` with the actual version number.
