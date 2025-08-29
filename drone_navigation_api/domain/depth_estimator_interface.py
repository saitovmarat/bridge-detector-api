from typing import Protocol
import numpy as np


class DepthEstimatorInterface(Protocol):
    def estimate(self, image: np.ndarray) -> np.ndarray:
        ...
