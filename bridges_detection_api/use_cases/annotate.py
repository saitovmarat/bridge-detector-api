import base64
import io
from PIL import Image
import numpy as np
import cv2
from typing import List
from ..domain.detection_result_dto import DetectionResultDTO 
from ..domain.image_dto import ImageDTO
from ..domain.annotated_image_dto import AnnotatedImageDTO


def annotated_image(
    img_dto: ImageDTO,
    detections: List[DetectionResultDTO]
) -> AnnotatedImageDTO:
    
    img = np.array(img_dto.pixels, dtype=np.uint8)
    annotated_img = img.copy()

    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    im_pil = Image.fromarray(annotated_img)

    img_io = io.BytesIO()
    im_pil.save(img_io, format='PNG')

    return AnnotatedImageDTO(image_data=base64.b64encode(img_io.getvalue()).decode())