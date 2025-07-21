import logging
from typing import List
import numpy as np
from ultralytics import YOLO
from ..domain.detection_result_dto import DetectionResultDTO
from ..domain.image_dto import ImageDTO


def detected_objects(
    img_dto: ImageDTO, 
    model: YOLO,
    threshold: float = 0.2
) -> List[DetectionResultDTO]:
    
    try:
        img_array = np.array(img_dto.pixels, dtype=np.uint8)
        if img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError("Ожидается трехмерный массив с 3 каналами (H x W x C).")
    except Exception as e:
        raise ValueError(f"Ошибка преобразования изображения в массив: {e}")
    
    results = model(img_array)
    
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf < threshold:
                    continue

                class_name = model.names[cls]

                detections.append(
                    DetectionResultDTO(
                        class_name=class_name,
                        confidence=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2
                    )
                )
            except KeyError as e:
                logging.error(f"Неизвестный ID класса: {e}")
                continue
            except Exception as e:
                logging.error(f"Неожиданная ошибка: {type(e).__name__}: {e}")
                continue

    return detections