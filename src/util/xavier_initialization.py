import numpy as np
from numpy.typing import NDArray


def xavier_initialization(rows: int, columns: int) -> NDArray[np.float64]:
    stddev = np.sqrt(2 / (rows + columns))

    return np.random.randn(rows, columns) * stddev
