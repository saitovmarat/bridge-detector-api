import io
from PIL import Image
import numpy as np
import pytest
from bridges_detection_api.infrastructure.img_processing import (
    load_image_from_bytes,
    image_to_array,
    rgb_to_bgr,
    array_to_dto
)
from bridges_detection_api.domain.image_dto import ImageDTO


@pytest.fixture
def test_image_stream():
    img = Image.new('RGB', (2, 2), color='red')
    
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    byte_io.seek(0)
    return byte_io


def test_load_image_from_bytes(test_image_stream):
    img = load_image_from_bytes(test_image_stream)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (2, 2)
    
    
def test_image_to_array(test_image_stream):
    img = load_image_from_bytes(test_image_stream)
    img_np = image_to_array(img)
    
    assert isinstance(img_np, np.ndarray)
    assert img_np.shape == (2, 2, 3)
    assert img_np.dtype == np.uint8
    
    
def test_rgb_to_bgr(test_image_stream):
    img = load_image_from_bytes(test_image_stream)
    img_np = image_to_array(img)
    img_bgr = rgb_to_bgr(img_np)
    
    assert img_bgr.shape == (2, 2, 3)
    assert img_bgr.dtype == np.uint8
    assert not np.array_equal(img_np, img_bgr)
    
    
def test_array_to_dto(test_image_stream):
    img = load_image_from_bytes(test_image_stream)
    img_np = image_to_array(img)
    img_bgr = rgb_to_bgr(img_np)
    dto = array_to_dto(img_bgr)
    
    assert isinstance(dto, ImageDTO)
    assert dto.width == 2
    assert dto.height == 2
    assert dto.format == "BGR"
    assert len(dto.pixels) == 2
    assert len(dto.pixels[0]) == 2
    assert len(dto.pixels[0][0]) == 3