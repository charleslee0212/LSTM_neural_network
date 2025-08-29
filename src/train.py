import sys
import numpy as np
from core.neural_network import LSTM_NeuralNetwork
from util.file_manager import loadFile, create_dataset, split_data

if __name__ == "__main__":
    epoch = 150
    batch_size = 32
    learning_rate = 0.00001
    decay = 1

    data = loadFile(filename="METR-LA.h5")

    time_step = 24
    input_size = data.shape[1]

    train, _, _, _ = split_data(data=data)

    in_seq, out_seq = create_dataset(data=train, t_in=time_step)

    neural_network = LSTM_NeuralNetwork(input_size=input_size, state_size=64)
    for e in range(epoch):
        idx = np.random.permutation(len(in_seq))
        in_seq = in_seq[idx]
        out_seq = out_seq[idx]
        for i in range(0, len(in_seq), batch_size):
            in_batch = in_seq[i : i + batch_size]
            out_batch = out_seq[i : i + batch_size]

            if in_batch.shape[0] != batch_size:
                continue

            in_series = np.transpose(in_batch, (1, 2, 0))
            out_series = np.transpose(out_batch, (1, 0))
            neural_network.forward_pass(series=in_series)
            neural_network.backpropagation(
                target=out_series, learning_rate=learning_rate
            )
            msg = f"Epoch: {e+1} | Training in progress: {i}/{len(in_seq)}"
            sys.stdout.write("\r\033[K" + msg)
        learning_rate *= decay

    neural_network.save()
