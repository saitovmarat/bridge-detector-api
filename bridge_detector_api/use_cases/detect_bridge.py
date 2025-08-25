from typing import List
from ..domain.detection_dto import DetectionDTO
from ..domain.image_dto import ImageDTO
from ..domain.detector_interface import DetectorInterface


def detect_bridge(
    img_dto: ImageDTO,
    detector: DetectorInterface
) -> List[DetectionDTO]:
    return detector.detect(img_dto.image)
