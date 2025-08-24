import base64
import numpy as np
from PIL import Image, ImageChops
from io import BytesIO
from bridge_detector_api.domain.image_dto import ImageDTO
from bridge_detector_api.domain.detection_dto import DetectionResultDTO
from bridge_detector_api.domain.annotated_image_dto import AnnotatedImageDTO
from bridge_detector_api.use_cases.annotate import annotated_image


def create_test_image(width=100, height=100):
    return Image.new("RGB", (width, height), "white")


def images_are_equal(img1: Image.Image, img2: Image.Image) -> bool:
    diff = ImageChops.difference(img1, img2)
    return not diff.getbbox()


def test_annotated_image_returns_valid_dto():
    pil_img = create_test_image(200, 200)
    img_np = np.array(pil_img)
    img_dto = ImageDTO(pixels=img_np.tolist(), width=200, height=200, format="RGB")

    detections = [
        DetectionResultDTO(class_name="car", confidence=0.95, x1=10, y1=10, x2=100, y2=100),
        DetectionResultDTO(class_name="person", confidence=0.85, x1=50, y1=50, x2=150, y2=150)
    ]

    result = annotated_image(img_dto, detections)

    assert isinstance(result, AnnotatedImageDTO)
    assert result.image_data is not None
    assert len(result.image_data) > 0


def test_annotated_image_draws_boxes_and_text():
    pil_img = create_test_image(200, 200)
    img_np = np.array(pil_img)
    img_dto = ImageDTO(pixels=img_np.tolist(), width=200, height=200, format="RGB")

    detections = [
        DetectionResultDTO(class_name="car", confidence=0.95, x1=10, y1=10, x2=100, y2=100),
        DetectionResultDTO(class_name="person", confidence=0.85, x1=50, y1=50, x2=150, y2=150)
    ]

    result = annotated_image(img_dto, detections)

    image_bytes = base64.b64decode(result.image_data)
    output_image = Image.open(BytesIO(image_bytes))

    difference = ImageChops.invert(output_image).getbbox()
    assert difference is not None  
    

def test_annotated_image_with_zero_detections():
    pil_img = create_test_image(200, 200)
    img_np = np.array(pil_img)
    img_dto = ImageDTO(pixels=img_np.tolist(), width=200, height=200, format="RGB")
    detections = []

    result = annotated_image(img_dto, detections)

    image_bytes = base64.b64decode(result.image_data)
    output_image = Image.open(BytesIO(image_bytes))

    original = Image.open(BytesIO(base64.b64decode(annotated_image(img_dto, []).image_data)))
    assert images_are_equal(original, output_image)


def test_annotated_image_with_negative_coordinates():
    pil_img = create_test_image(200, 200)
    img_np = np.array(pil_img)
    img_dto = ImageDTO(pixels=img_np.tolist(), width=200, height=200, format="RGB")

    detections = [
        DetectionResultDTO(class_name="car", confidence=0.95, x1=-10, y1=-10, x2=50, y2=50)
    ]

    result = annotated_image(img_dto, detections)

    assert isinstance(result, AnnotatedImageDTO)
    assert result.image_data is not None


def test_annotated_image_invalid_bbox_does_not_crash():
    pil_img = create_test_image(200, 200)
    img_np = np.array(pil_img)
    img_dto = ImageDTO(pixels=img_np.tolist(), width=200, height=200, format="RGB")

    detections = [
        DetectionResultDTO(class_name="car", confidence=0.95, x1=10, y1=10, x2=300, y2=300)
    ]

    result = annotated_image(img_dto, detections)

    assert isinstance(result, AnnotatedImageDTO)
    assert result.image_data is not None


def test_annotated_image_decodes_back_to_image():
    pil_img = create_test_image(200, 200)
    img_np = np.array(pil_img)
    img_dto = ImageDTO(pixels=img_np.tolist(), width=200, height=200, format="RGB")
    detections = []

    result = annotated_image(img_dto, detections)

    decoded = base64.b64decode(result.image_data)
    image = Image.open(BytesIO(decoded))

    assert image.size == (200, 200)
    assert image.mode == 'RGB'