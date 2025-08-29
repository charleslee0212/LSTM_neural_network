import numpy as np
from numpy.typing import NDArray
import h5py
import os


def split_data(
    data: NDArray,
    train_ratio: float = 0.8,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    num_samples = data.shape[0]
    train_size = int(num_samples * train_ratio)

    data_train = data[:train_size]
    data_test = data[train_size:]

    # normalization
    mean = data_train.mean(axis=0)
    std = data_train.std(axis=0)

    std[std == 0] = 1e-8

    data_train = (data_train - mean) / std
    data_test = (data_test - mean) / std

    return data_train, data_test, mean, std


def create_dataset(
    data: NDArray[np.float64],
    t_in: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    time_steps, sensors = data.shape
    sample_size = time_steps - t_in

    x: NDArray[np.float64] = np.zeros((sample_size, t_in, sensors))
    y: NDArray[np.float64] = np.zeros((sample_size, sensors))

    for i in range(sample_size):
        x[i] = data[i : i + t_in]
        y[i] = data[i + t_in]

    return x, y


def loadFile(filename: str) -> NDArray[np.float64]:
    dir = "./data"
    file_path = os.path.join(dir, filename)
    with h5py.File(file_path, "r") as f:

        data = f["df"]["block0_values"][:]
        return data
