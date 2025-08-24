from dataclasses import dataclass


@dataclass
class AnnotatedImageDTO:
    image_data: str 
    image_format: str = "RGB"  
