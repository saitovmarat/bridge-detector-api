from numpy.typing import NDArray
from typing import List
from ultralytics import YOLO
from bridges_detection_api.domain.detection_result import DetectionResult


def detected_objects(img_bgr: NDArray, model: YOLO) -> List[DetectionResult]:
    results = model(img_bgr)
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.2:
                detection = DetectionResult(
                    class_name=model.names[cls],
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                )
                detections.append(detection)

    return detections