import socket
import signal
import threading
from typing import Optional

from ..use_cases.process_udp_packet import process_packet
from ..utils.model_loader import load_model
from ..infrastructure.depth_estimator import DepthEstimator

_shutdown_event = threading.Event()
_cache = {}


def _signal_handler(signum, frame):
    """Обработчик сигналов для остановки сервера"""
    print(
        f"\n🛑 Получен сигнал {signal.Signals(signum).name}. Останавливаю сервер...")
    _shutdown_event.set()


def run_udp_server(host: str = "0.0.0.0", port: int = 9999):
    """
    Запускает UDP-сервер для приёма кадров, детекции мостов и отправки результатов.
    Сервер можно остановить через Ctrl+C
    """
    signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C

    detection_model = load_model()
    depth_estimator = DepthEstimator()

    print("🧠 Модель загружена")

    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        print(f"🚀 UDP API запущен на {host}:{port}")

        while not _shutdown_event.is_set():
            sock.settimeout(1.0)
            try:
                data, addr = sock.recvfrom(65507)
                print(f"📩 Получено {len(data)} байт от {addr}")

                response = process_packet(
                    data, detection_model, depth_estimator, cache=_cache)
                sock.sendto(response, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if _shutdown_event.is_set():
                    break
                print(f"❌ Ошибка приёма: {e}")

    except Exception as e:
        print(f"❌ Ошибка сервера: {e}")
    finally:
        if sock:
            sock.close()
        print("🛑 UDP сервер остановлен. Ресурсы освобождены.")
        _shutdown_event.clear()
