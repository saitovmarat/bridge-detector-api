from ultralytics import YOLO
from typing import List
from ..domain.detection_dto import DetectionDTO


class YoloDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image) -> List[DetectionDTO]:
        results = self.model.track(source=image, persist=True, verbose=False)
        if not results:
            return []

        detections = []
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return detections

        xyxy_tensor = boxes.xyxy
        conf_tensor = boxes.conf
        id_tensor = boxes.id

        xyxy = xyxy_tensor.cpu().numpy()    # type: ignore
        conf = conf_tensor.cpu().numpy()    # type: ignore
        track_ids = id_tensor.cpu().numpy()  # type: ignore

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            detection = DetectionDTO(
                class_name="bridge",
                confidence=float(conf[i]),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                track_id=int(track_ids[i])
            )
            detections.append(detection)

        return detections
