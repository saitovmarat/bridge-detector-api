import socket
import json
import base64
import io
import signal
import threading
from typing import Optional

from bridges_detection_api.use_cases.img_preprocessor import preprocess_image_dto
from bridges_detection_api.use_cases.detect import detected_objects
from bridges_detection_api.utils.model_loader import load_model


_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    """Обработчик сигналов для остановки сервера"""
    print(f"\n🛑 Получен сигнал {signal.Signals(signum).name}. Останавливаю сервер...")
    _shutdown_event.set()


def run_udp_server(host: str = "0.0.0.0", port: int = 9999):
    """
    Запускает UDP-сервер для приёма кадров, детекции мостов и отправки результатов.
    Сервер можно остановить через Ctrl+C
    """
    signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C

    model = load_model()
    sock: Optional[socket.socket] = None

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        print(f"🚀 UDP API запущен на {host}:{port}")
        print("💡 Для остановки сервера нажмите Ctrl+C")

        while not _shutdown_event.is_set():
            sock.settimeout(1.0)  # 1 секунда
            try:
                data, addr = sock.recvfrom(65507)  # Макс UDP-пакет
                print(f"📩 Получено {len(data)} байт от {addr}")

                try:
                    packet = json.loads(data.decode('utf-8'))
                    b64_image = packet.get("frame")
                    if not b64_image:
                        error = json.dumps({"error": "No 'frame' in request"}).encode()
                        sock.sendto(error, addr)
                        continue

                    image_bytes = base64.b64decode(b64_image)
                    img_stream = io.BytesIO(image_bytes)
                except Exception as e:
                    error = json.dumps({
                        "error": "Invalid request format",
                        "details": str(e)
                    }).encode()
                    sock.sendto(error, addr)
                    continue

                try:
                    img_dto = preprocess_image_dto(img_stream)
                    detections = detected_objects(img_dto, model)

                    response = {
                        "success": True,
                        "detections": [
                            {
                                "class": det.class_name,
                                "confidence": round(det.confidence, 3),
                                "x1": int(det.x1),
                                "y1": int(det.y1),
                                "x2": int(det.x2),
                                "y2": int(det.y2)
                            }
                            for det in detections
                        ]
                    }
                    sock.sendto(json.dumps(response).encode('utf-8'), addr)
                except Exception as e:
                    error = json.dumps({
                        "error": "Processing failed",
                        "details": str(e)
                    }).encode()
                    sock.sendto(error, addr)

            except socket.timeout:
                continue
            except Exception as e:
                if _shutdown_event.is_set():
                    break 
                print(f"❌ Ошибка приёма: {e}")

    except Exception as e:
        print(f"❌ Ошибка сервера: {e}")
    finally:
        if sock:
            sock.close()
        print("🛑 UDP сервер остановлен. Ресурсы освобождены.")
        _shutdown_event.clear()  