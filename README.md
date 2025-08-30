# 🌉 drone-navigation-api — Real-Time Bridge, Arch Detection & Navigation Target Generator

**drone-navigation-api** is a Python-based AI server for real-time detection of **bridges**, **arches**, and generation of **navigation targets** in video and image streams. Designed for drone-based infrastructure inspection, it powers intelligent flight paths by combining object detection with spatial reasoning.

The API processes frames sent from clients like **[drone-navigation-client](https://github.com/saitovmarat/drone-navigation-client)** and returns structured detection results — enabling autonomous drones to **see**, **understand**, and **navigate** complex bridge structures.

---

## 🚀 Features

### ✅ Multi-Object Detection
- 🔲 **Bridges**: Full structure localization
- 🏛️ **Arches**: Detection of individual arch elements within bridges
- 🎯 **Navigation Targets**: Automatically generated waypoints for drone inspection routes

### ⚙️ Dual Operation Modes
- **UDP Mode** — ultra-low-latency streaming for real-time drone video
- **REST Mode** — flexible integration for batch processing or web apps

### 🛑 Graceful Shutdown
Supports clean exit via `Ctrl+C` or `SIGTERM` — ideal for Docker, Kubernetes, and production environments.

### 📦 Ready-to-Deploy
Distributed as a standalone `.whl` package — install and run in seconds.

---

## 🧪 Technologies Used
- **Python 3.8+** — Core runtime
- **Ultralytics YOLO** — High-speed object detection (custom-trained)
- **Flask** — REST API backend
- **OpenCV, Pillow, NumPy** — Image preprocessing and encoding
- **UDP/JSON/base64** — Lightweight, real-time communication with clients

---

## 📦 Installation

1. **Download the latest `.whl` package** from the [releases page](https://github.com/saitovmarat/drone-navigation-api/releases)
2. **Install via pip** (recommended in a virtual environment):
   ```bash
   pip install drone_navigation_api-X.X.X-py3-none-any.whl
