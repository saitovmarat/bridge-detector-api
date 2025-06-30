from dataclasses import dataclass


@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int