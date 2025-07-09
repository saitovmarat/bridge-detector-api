from typing import IO
import numpy as np
from PIL import Image
import cv2
from ..domain.image_dto import ImageDTO


def load_image_from_bytes(image_stream: IO[bytes]) -> Image.Image:
    return Image.open(image_stream).convert("RGB")


def image_to_array(img: Image.Image) -> np.ndarray:
    return np.array(img)


def rgb_to_bgr(img_np: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def array_to_dto(img_np: np.ndarray) -> ImageDTO:
    h, w, c = img_np.shape
    return ImageDTO(
        pixels=img_np.tolist(),
        width=w,
        height=h,
        format="BGR"
    )