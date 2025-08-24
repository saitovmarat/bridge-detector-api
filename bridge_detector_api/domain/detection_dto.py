from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionDTO:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None
