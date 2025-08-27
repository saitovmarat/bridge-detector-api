from typing import Any, Dict


def is_bridge_confirmed(cache: Dict[str, Any]) -> bool:
    history = cache.get("bridge_detection_history", [])
    if len(history) < 5:
        return False

    recent = history[-5:]
    return sum(recent) >= 4


# Разные реализации метода
# ==============================
#
# С track_id (Если на изображении 5 раз подряд ОДИН И ТОТ ЖЕ мост)
# def is_bridge_confirmed(cache):
#     track_id_streak = cache.get("track_id_streak", 0)
#     return track_id_streak >= 5
#
# Проверка, что мост не прыгает
# def is_bridge_confirmed(cache):
#     history = cache.get("bridge_detection_history", [])
#     positions = cache.get("bridge_positions", [])

#     if len(history) < 5 or sum(history) < 5:
#         return False

#     if len(positions) >= 2:
#         xs = [pos[0] for pos in positions[-5:]]
#         ys = [pos[1] for pos in positions[-5:]]
#         std_x = np.std(xs)
#         std_y = np.std(ys)
#         return std_x < 20 and std_y < 15

#     return True
