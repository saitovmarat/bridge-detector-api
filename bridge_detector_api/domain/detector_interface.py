from typing import Protocol, List
import numpy as np

from .detection_dto import DetectionDTO


class DetectorInterface(Protocol):
    def detect(self, image: np.ndarray) -> List[DetectionDTO]:
        ...
