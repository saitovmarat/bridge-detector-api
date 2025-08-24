import json
import base64
import pytest
import numpy as np
import cv2

from bridge_detector_api.use_cases.process_udp_packet import process_packet


@pytest.fixture
def mock_detection_model():
    class MockModel:
        def __init__(self):
            self.names = {0: "bridge"} 

        def __call__(self, image_array):
            class MockBox:
                def __init__(self):
                    self.xyxy = np.array([[100, 100, 500, 300]])
                    self.conf = np.array([0.95])
                    self.cls = np.array([0])

            class MockBoxes:
                def __init__(self):
                    self.data = [MockBox()]

                def __iter__(self):
                    return iter(self.data)

            class MockResult:
                def __init__(self):
                    self.boxes = MockBoxes()
                    self.orig_shape = image_array.shape[:2]

            return [MockResult()]

    return MockModel()


@pytest.fixture
def mock_depth_estimator():
    class MockDepthEstimator:
        def estimate(self, image):
            h, w = image.shape[:2]
            return np.random.rand(h, w).astype(np.float32)

    return MockDepthEstimator()


@pytest.fixture
def sample_image_bytes():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), 2)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def test_process_packet_detect_mode(
    sample_image_bytes, mock_detection_model, mock_depth_estimator
):
    b64_img = base64.b64encode(sample_image_bytes).decode('utf-8')
    packet = json.dumps({"mode": "detect", "frame": b64_img})
    data = packet.encode('utf-8')

    response = process_packet(data, mock_detection_model, mock_depth_estimator)
    result = json.loads(response.decode('utf-8'))

    assert result["success"] is True
    assert len(result["detections"]) == 1
    det = result["detections"][0]
    assert det["class"] == "bridge"
    assert det["confidence"] == 0.95
    assert det["x1"] == 100
    assert det["y1"] == 100
    assert det["x2"] == 500
    assert det["y2"] == 300
    assert "depth_maps" not in result

def test_process_packet_detect_depth_mode(
    sample_image_bytes, mock_detection_model, mock_depth_estimator
):
    b64_img = base64.b64encode(sample_image_bytes).decode('utf-8')
    packet = json.dumps({"mode": "detect+depth", "frame": b64_img})
    data = packet.encode('utf-8')

    response = process_packet(data, mock_detection_model, mock_depth_estimator)
    result = json.loads(response.decode('utf-8'))

    assert result["success"] is True
    assert "depth_maps" in result
    assert len(result["depth_maps"]) == 1

    depth = result["depth_maps"][0]
    assert depth["bbox"]["x1"] == 100
    assert depth["bbox"]["y1"] == 100
    assert depth["bbox"]["x2"] == 500
    assert depth["bbox"]["y2"] == 300
    assert depth["confidence"] == 0.95
    assert "depth_map" in depth
    assert isinstance(depth["depth_map"], str)
    assert len(depth["depth_map"]) > 100