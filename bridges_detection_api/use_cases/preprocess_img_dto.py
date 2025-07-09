from typing import IO
from ..infrastructure.img_processing import \
    array_to_dto, load_image_from_bytes, image_to_array
from ..domain.image_dto import ImageDTO


def preprocess_image_dto(image_stream: IO[bytes]) -> ImageDTO:
    pil_img = load_image_from_bytes(image_stream)
    img_np = image_to_array(pil_img)
    return array_to_dto(img_np)