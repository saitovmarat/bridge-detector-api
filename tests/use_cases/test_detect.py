import numpy as np
from unittest.mock import MagicMock, Mock

import pytest

from bridge_detector_api.domain.image_dto import ImageDTO
from bridge_detector_api.domain.detection_dto import DetectionResultDTO
from bridge_detector_api.use_cases.detect_bridge import detected_objects


def create_test_image_array(height=100, width=100):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


class MockBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class MockBoxes:
    def __init__(self, boxes):
        self.boxes = boxes

    def __iter__(self):
        return iter(self.boxes)


class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_detected_objects_valid_image():
    img_array = create_test_image_array()
    img_dto = ImageDTO(pixels=img_array.tolist(), width=100, height=100, format="RGB")
    model = MagicMock()
    model.names = {0: "car", 1: "person"}

    mock_boxes = [
        MockBox(xyxy=[[10, 10, 50, 50]], conf=[0.9], cls=[0]),
        MockBox(xyxy=[[60, 60, 100, 100]], conf=[0.7], cls=[1])
    ]
    results = [MockResult(MockBoxes(mock_boxes))]
    model.return_value = results

    detections = detected_objects(img_dto, model)

    assert isinstance(detections, list)
    assert len(detections) == 2

    assert detections[0].class_name == "car"
    assert detections[0].confidence == 0.9
    assert detections[0].x1 == 10
    assert detections[0].y1 == 10
    assert detections[0].x2 == 50
    assert detections[0].y2 == 50

    assert detections[1].class_name == "person"
    assert detections[1].confidence == 0.7
    assert detections[1].x1 == 60
    assert detections[1].y1 == 60
    assert detections[1].x2 == 100
    assert detections[1].y2 == 100


def test_detected_objects_invalid_shape():
    img_array = create_test_image_array()
    img_dto = ImageDTO(
        pixels=img_array[:, :, :1].tolist(),
        width=100,
        height=100,
        format="RGB"
    )
    model = MagicMock()

    with pytest.raises(ValueError):
        detected_objects(img_dto, model)


def test_detected_objects_empty_detections():
    img_array = create_test_image_array()
    img_dto = ImageDTO(pixels=img_array.tolist(), width=100, height=100, format="RGB")
    model = MagicMock()
    model.names = {0: "car"}

    results = [MagicMock()]
    results[0].boxes = [] 
    model.return_value = results

    detections = detected_objects(img_dto, model)

    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detected_objects_below_threshold():
    img_array = create_test_image_array()
    img_dto = ImageDTO(pixels=img_array.tolist(), width=100, height=100, format="RGB")
    model = MagicMock()
    model.names = {0: "car"}

    mock_box = MockBox(xyxy=[[10, 10, 50, 50]], conf=[0.15], cls=[0])  # ниже порога
    results = [MockResult(MockBoxes([mock_box]))]
    model.return_value = results

    detections = detected_objects(img_dto, model, threshold=0.2)

    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detected_objects_with_invalid_class_id(caplog):
    img_array = create_test_image_array()
    img_dto = ImageDTO(pixels=img_array.tolist(), width=100, height=100, format="RGB")
    model = MagicMock()
    model.names = {0: "car"} 

    mock_box = MockBox(xyxy=[[10, 10, 50, 50]], conf=[0.9], cls=[1]) 
    results = [MockResult(MockBoxes([mock_box]))]
    model.return_value = results

    detections = detected_objects(img_dto, model)

    assert isinstance(detections, list)
    assert len(detections) == 0  
    assert any("Неизвестный ID класса" in record.message for record in caplog.records)
    

def test_detected_objects_with_exception_in_box_processing(caplog):
    img_array = create_test_image_array()
    img_dto = ImageDTO(pixels=img_array.tolist(), width=100, height=100, format="RGB")
    model = MagicMock()
    model.names = {0: "car"}

    class MockBoxWithError:
        @property
        def xyxy(self):
            raise Exception("Ошибка при получении координат")

        @property
        def conf(self):
            return [[0.9]]

        @property
        def cls(self):
            return [[0]]

    mock_box = MockBoxWithError()
    results = [MockResult(MockBoxes([mock_box]))]
    model.return_value = results

    detections = detected_objects(img_dto, model)

    assert isinstance(detections, list)
    assert len(detections) == 0
    assert any("Неожиданная ошибка" in record.message for record in caplog.records)