import cv2
import numpy as np
from io import BufferedIOBase
from ..domain.image_dto import ImageDTO


def preprocess_image_dto(img_stream: BufferedIOBase) -> ImageDTO:
    image = cv2.imdecode(np.frombuffer(
        img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    return ImageDTO(pixels=image, image=image)
