def update_bridge_confirmation(
    detections: list,
    cache: dict,
    max_history: int = 10
) -> None:
    has_detections = len(detections) > 0

    history = cache.get("bridge_detection_history", [])
    history.append(has_detections)

    if len(history) > max_history:
        history = history[-max_history:]

    cache["bridge_detection_history"] = history
