from typing import IO
import numpy as np
from PIL import Image
import cv2


def preprocess_image(image_bytes: IO[bytes]) -> np.ndarray: 
    img = Image.open(image_bytes).convert('RGB')
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr