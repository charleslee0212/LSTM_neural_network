import numpy as np
from numpy.typing import NDArray


def clip_gradients(delta: NDArray[np.float64], threshold=5.0) -> NDArray[np.float64]:
    return np.clip(delta, -threshold, threshold)
