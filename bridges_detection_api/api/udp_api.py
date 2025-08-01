import socket
import json
import base64
import io

from bridges_detection_api.use_cases.img_preprocessor import preprocess_image_dto
from bridges_detection_api.use_cases.detect import detected_objects
from bridges_detection_api.utils.model_loader import load_model 


def run_udp_server(host="0.0.0.0", port=9999):
    model = load_model()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"🚀 UDP API запущен на {host}:{port}")

    while True:
        try:
            data, addr = sock.recvfrom(65507)  # Макс UDP-пакет 65 Кбайт
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

        except KeyboardInterrupt:
            print("\n🛑 UDP сервер остановлен.")
            break
        except Exception as e:
            print(f"❌ Ошибка сервера: {e}")