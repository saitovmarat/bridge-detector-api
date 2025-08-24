from dataclasses import dataclass
from typing import Optional

@dataclass
class ArchCenterDTO:
    x: int
    y: int
    source_detection: Optional[int] = None 