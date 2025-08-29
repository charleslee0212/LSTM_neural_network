import numpy as np
import yfinance as yf
from core.neural_network import LSTM_NeuralNetwork
from util.file_manager import loadFile, create_dataset, split_data

if __name__ == "__main__":
    data = loadFile(filename="METR-LA.h5")

    time_step = 12
    input_size = data.shape[1]
    batch_size = 1

    _, test, mean, std = split_data(data=data)
    in_seq, out_seq = create_dataset(data=test, t_in=time_step)

    neural_network = LSTM_NeuralNetwork.load("model.pkl")

    cost = 0

    for i in range(len(in_seq)):
        in_series = in_seq[i].reshape(time_step, input_size, batch_size)
        out_series = out_seq[i].reshape(input_size, batch_size)
        neural_network.forward_pass(series=in_series)

        prediction = neural_network.output

        # denormalize
        prediction_denorm = prediction * std + mean
        out_series_denorm = out_series * std + mean

        cost += np.mean(np.square(out_series_denorm - prediction_denorm))
        print(
            f"Testing in progress: {i+1}/{len(in_seq)}",
            end="\r",
        )

    print()
    rmse = np.sqrt(cost / len(in_seq))
    print("RMSE:", rmse)
