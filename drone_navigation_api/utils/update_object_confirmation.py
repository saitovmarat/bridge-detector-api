def update_object_confirmation(
    object_name: str,
    detections: list,
    cache: dict,
    max_history: int = 10
) -> None:
    
    has_detections = len(detections) > 0
    history = cache.get(f"{object_name}_detection_history", [])
    history.append(has_detections)

    if len(history) > max_history:
        history = history[-max_history:]

    cache[f"{object_name}_detection_history"] = history
