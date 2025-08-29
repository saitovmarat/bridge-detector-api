from dataclasses import dataclass


@dataclass
class DetectionDTO:
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int

    def to_dict(self):
        return {
            "confidence": round(self.confidence, 3),
            "x1": int(self.x1),
            "y1": int(self.y1),
            "x2": int(self.x2),
            "y2": int(self.y2),
            "track_id": self.track_id
        }
