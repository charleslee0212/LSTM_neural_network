import numpy as np
from numpy.typing import NDArray


def _nan_guard(arr: NDArray[np.float64], name: str = "array") -> None:
    if np.any(np.isnan(arr)):
        raise ValueError(f"NaN detected in {name}")
    if np.any(np.isinf(arr)):
        raise ValueError(f"Inf detected in {name}")


def guard_output(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        _nan_guard(result, func.__name__)
        return result

    return wrapper
