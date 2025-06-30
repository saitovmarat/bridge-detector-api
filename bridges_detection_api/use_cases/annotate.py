from numpy.typing import NDArray
import cv2
from typing import List
from bridges_detection_api.domain.detection_result import DetectionResult  


def annotated_image(img_bgr: NDArray, detections: List[DetectionResult]) -> NDArray:
    annotated_img = img_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_img_rgb