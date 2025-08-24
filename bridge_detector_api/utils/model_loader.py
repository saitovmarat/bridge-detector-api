from importlib.resources import files, as_file
from ..infrastructure.yolo_detector import YoloDetector

def load_model():
    ref = files("bridge_detector_api.assets") / "best_weights.pt"
    with as_file(ref) as model_path:
        return YoloDetector(str(model_path))
