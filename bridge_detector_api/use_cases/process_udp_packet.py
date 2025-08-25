import json
import base64
import io

from .detect_bridge import detect_bridge
from .find_arch_center import find_arch_center
from .img_preprocessor import preprocess_image_dto


def process_packet(
    data: bytes,
    detector,
    depth_estimator,
    cache: dict,
    depth_update_every: int = 5
) -> bytes:
    """
    Use Case: обработать UDP-пакет.
    Возвращает JSON с детекциями.
    """
    try:
        packet = json.loads(data.decode('utf-8'))
        b64_image = packet.get("frame")
        frame_id = packet.get("frame_id", 0)

        if not b64_image:
            return json.dumps({"error": "No 'frame' in request"}).encode()
        image_bytes = base64.b64decode(b64_image)
        img_stream = io.BytesIO(image_bytes)
    except Exception as e:
        return json.dumps({"error": "Invalid format", "details": str(e)}).encode()

    try:
        img_dto = preprocess_image_dto(img_stream)
        image = img_dto.image

        detections = detect_bridge(img_dto, detector)

        arch_center = find_arch_center(
            image=image,
            detections=detections,
            depth_estimator=depth_estimator,
            frame_id=frame_id,
            depth_update_every=depth_update_every,
            cache=cache
        )

        response = {
            "success": True,
            "detections": [
                {
                    "class": det.class_name,
                    "confidence": round(det.confidence, 3),
                    "x1": int(det.x1),
                    "y1": int(det.y1),
                    "x2": int(det.x2),
                    "y2": int(det.y2),
                    "track_id": det.track_id
                }
                for det in detections
            ]
        }

        if arch_center:
            response["arch_center"] = {
                "x": arch_center.x,
                "y": arch_center.y
            }

        return json.dumps(response).encode('utf-8')

    except Exception as e:
        return json.dumps({"error": "Processing failed", "details": str(e)}).encode()
