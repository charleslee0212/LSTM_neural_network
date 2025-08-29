import numpy as np
from numpy.typing import NDArray


def create_datasets(data: NDArray, time_step: int, train_ratio: float = 0.8):
    eps = 1e-8
    log_returns = np.diff(np.log(data + eps), axis=0)

    train_size = int(len(log_returns) * train_ratio)
    train_data = log_returns[:train_size]
    test_data = log_returns[train_size:]

    # normalize based only on training stats
    min_p = np.min(train_data)
    max_p = np.max(train_data)

    train_data = (train_data - min_p) / (max_p - min_p)
    test_data = (test_data - min_p) / (max_p - min_p)

    def make_xy(series):
        X, y = [], []
        for i in range(len(series) - time_step):
            X.append(series[i : i + time_step])
            y.append(series[i + time_step])
        return np.array(X), np.array(y)

    X_train, y_train = make_xy(train_data)
    X_test, y_test = make_xy(test_data)

    return X_train, y_train, X_test, y_test
