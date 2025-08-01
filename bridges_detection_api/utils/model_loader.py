from ultralytics import YOLO
from importlib.resources import files, as_file

def load_model():
    ref = files("bridges_detection_api.assets") / "best_weights.pt"
    with as_file(ref) as model_path:
        return YOLO(str(model_path))