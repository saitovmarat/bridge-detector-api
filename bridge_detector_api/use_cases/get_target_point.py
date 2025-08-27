import cv2
import numpy as np
from typing import Optional, List
from ..domain.detection_dto import DetectionDTO
from ..domain.arch_center_dto import ArchCenterDTO
from ..domain.depth_estimator_interface import DepthEstimatorInterface


def get_target_point(
    image,
    depth_estimator: DepthEstimatorInterface,
    arch_gap_detections: List[DetectionDTO],
    min_roi_size: tuple[int, int] = (10, 10)  # можно убрать или изменить
) -> Optional[ArchCenterDTO]:

    if len(arch_gap_detections) == 0:
        return None

    h, w = image.shape[:2]

    # берем только одну детекцию - первую
    # TODO: изменить логику
    det = arch_gap_detections[0]

    try:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
    except (ValueError, TypeError):
        return None

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x1 >= x2 or y1 >= y2:
        return None

    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < min_roi_size[0] or roi.shape[1] < min_roi_size[1]:
        print(f"⚠️ ROI слишком мал: {roi.shape}")
        return None

    try:
        depth_map = depth_estimator.estimate(roi)
        if depth_map is None or depth_map.size == 0:
            return None

        depth_map = cv2.bilateralFilter(
            depth_map, d=9, sigmaColor=75, sigmaSpace=75)
        farthest_idx = np.unravel_index(np.argmin(depth_map), depth_map.shape)

        global_x = x1 + farthest_idx[1]
        global_y = y1 + farthest_idx[0]

        return ArchCenterDTO(
            x=int(global_x),
            y=int(global_y),
            source_detection=det.track_id
        )

    except Exception as e:
        print(f"❌ Ошибка при оценке глубины: {e}")
        return None
