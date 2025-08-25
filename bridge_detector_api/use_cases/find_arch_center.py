import cv2
import numpy as np
from collections import deque
from typing import Optional, List
from ..domain.detection_dto import DetectionDTO
from ..domain.arch_center_dto import ArchCenterDTO
from ..domain.depth_estimator_interface import DepthEstimatorInterface


_roi_buffer = {}


def find_arch_center(
    image,
    detections: List[DetectionDTO],
    depth_estimator: DepthEstimatorInterface,
    frame_id: int,
    cache: dict,
    depth_update_every: int = 5,
    min_roi_width: int = 32,
    min_roi_height: int = 16,
    buffer_size: int = 5
) -> Optional[ArchCenterDTO]:
    global _roi_buffer

    if image is None or not hasattr(image, 'shape'):
        return None

    if cache is None:
        cache = {}

    h, w = image.shape[:2]

    for det in detections:
        track_id = det.track_id

        try:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        except (ValueError, TypeError):
            continue

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x1 >= x2 or y1 >= y2:
            continue

        raw_roi_w = x2 - x1
        raw_roi_h = y2 - y1

        if track_id not in _roi_buffer:
            _roi_buffer[track_id] = {
                "widths": deque(maxlen=buffer_size),
                "heights": deque(maxlen=buffer_size)
            }

        buf = _roi_buffer[track_id]
        buf["widths"].append(raw_roi_w)
        buf["heights"].append(raw_roi_h)

        roi_y1 = y1 + (y2 - y1) // 2
        roi_y2 = min(y2 + 80, h)
        roi_x1 = max(0, x1)
        roi_x2 = min(w, x2)

        if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
            print(f"⚠️ Пропущен объединённый ROI: track_id={track_id}")
            continue

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0 or roi.shape[0] < min_roi_height or roi.shape[1] < min_roi_width:
            print(f"⚠️ ROI слишком мал: {roi.shape}, track_id={track_id}")
            continue

        should_update = (frame_id % depth_update_every ==
                         0) or (track_id not in cache)

        if not should_update and track_id in cache:
            cached = cache[track_id]
            return ArchCenterDTO(x=cached["x"], y=cached["y"], source_detection=track_id)

        try:
            depth_map = depth_estimator.estimate(roi)
            if depth_map is None or depth_map.size == 0:
                continue

            depth_map = cv2.bilateralFilter(
                depth_map, d=9, sigmaColor=75, sigmaSpace=75)
            min_idx = np.unravel_index(np.argmin(depth_map), depth_map.shape)

            arch_x = roi_x1 + min_idx[1]
            arch_y = roi_y1 + min_idx[0]

            arch_center = ArchCenterDTO(x=int(arch_x), y=int(
                arch_y), source_detection=track_id)
            cache[track_id] = {"x": arch_x, "y": arch_y}

            print(f"🟢 Центр арки найден: x={arch_x}, y={arch_y}")
            return arch_center

        except Exception as e:
            print(f"❌ Ошибка при оценке глубины для track_id={track_id}: {e}")
            continue

    return None
