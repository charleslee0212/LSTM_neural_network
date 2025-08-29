import numpy as np
from numpy.typing import NDArray
from .nan_guard import guard_output


@guard_output
def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.clip(x, -500, 500)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


@guard_output
def derivative_sigmoid(x: float) -> float:
    s = sigmoid(x)
    return s * (1 - s)


@guard_output
def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)


@guard_output
def derivative_tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 - tanh(x) ** 2
