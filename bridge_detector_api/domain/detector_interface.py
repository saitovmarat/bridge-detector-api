from typing import Protocol, List
from .image_dto import ImageDTO
from .detection_dto import DetectionDTO


class DetectorInterface(Protocol):
    def detect(self, image: ImageDTO) -> List[DetectionDTO]:
        raise NotImplementedError
