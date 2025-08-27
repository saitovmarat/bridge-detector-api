from dataclasses import dataclass
from typing import Optional


@dataclass
class ArchCenterDTO:
    x: int
    y: int
    source_detection: Optional[int] = None

    def to_dict(self):
        return {
            "x": int(self.x),
            "y": int(self.y),
            "source_detection": self.source_detection
        }
