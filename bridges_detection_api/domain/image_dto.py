from dataclasses import dataclass
from typing import List


@dataclass
class ImageDTO:
    pixels: List[List[List[int]]] 
    width: int
    height: int
    format: str = "BGR" 