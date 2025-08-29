import numpy as np
from numpy.typing import NDArray
import pickle
import os
from util.xavier_initialization import xavier_initialization
from util.functions import sigmoid, derivative_sigmoid, tanh, derivative_tanh
from util.clip_gradients import clip_gradients


class LSTM_NeuralNetwork:
    def __init__(self, input_size: int = 1, state_size: int = 64, batch: int = 1):
        self.input_size = input_size
        self.state_size = state_size
        self.batch = batch
        self.cell_state: NDArray[np.float64] = np.zeros((state_size, batch))
        self.hidden_state: NDArray[np.float64] = np.zeros((state_size, batch))
        self.weights: NDArray[np.float64] = xavier_initialization(
            4 * state_size, input_size + state_size
        )
        biases = np.zeros((4 * state_size, 1))
        biases[0:state_size] = 1.0
        self.biases: NDArray[np.float64] = biases

        self.weight_output: NDArray[np.float64] = xavier_initialization(
            input_size, state_size
        )
        self.bias_output: NDArray[np.float64] = np.zeros((input_size, 1))
        self.output: NDArray[np.float64] = np.array([])
        self.cache = []

    def forward_pass(self, series: NDArray[np.float64]) -> None:
        # series shape: (sequence_length, input_size, batch_size)
        seq_len, input_size, batch_size = series.shape
        assert input_size == self.input_size
        self.cell_state: NDArray[np.float64] = np.zeros((self.state_size, batch_size))
        self.hidden_state: NDArray[np.float64] = np.zeros((self.state_size, batch_size))
        self.cache.clear()
        for t in range(seq_len):
            x_t = series[t]
            concat = np.vstack((x_t, self.hidden_state))
            z = (self.weights @ concat) + self.biases

            forget_gate_z = z[0 : self.state_size]
            input_gate_z = z[self.state_size : 2 * self.state_size]
            candidate_gate_z = z[2 * self.state_size : 3 * self.state_size]
            output_gate_z = z[3 * self.state_size : 4 * self.state_size]

            forget_gate = sigmoid(z[0 : self.state_size])
            input_gate = sigmoid(z[self.state_size : 2 * self.state_size])
            candidate_gate = tanh(z[2 * self.state_size : 3 * self.state_size])
            output_gate = sigmoid(z[3 * self.state_size : 4 * self.state_size])

            cell_state_t = (forget_gate * self.cell_state) + (
                input_gate * candidate_gate
            )
            hidden_state_t = output_gate * tanh(cell_state_t)

            self.cache.append(
                {
                    "x_t": x_t,
                    "concat": concat,
                    "f_t": forget_gate,
                    "i_t": input_gate,
                    "g_t": candidate_gate,
                    "o_t": output_gate,
                    "c_t": cell_state_t,
                    "h_t": hidden_state_t,
                    "c_prev": self.cell_state,
                    "h_prev": self.hidden_state,
                    "z_f": forget_gate_z,
                    "z_i": input_gate_z,
                    "z_g": candidate_gate_z,
                    "z_o": output_gate_z,
                }
            )
            self.cell_state = cell_state_t
            self.hidden_state = hidden_state_t

        self.output = (self.weight_output @ self.hidden_state) + self.bias_output

    def backpropagation(
        self, target: NDArray[np.float64], learning_rate: float
    ) -> None:
        assert target.shape == self.output.shape and len(self.cache) > 0

        cost_delta: NDArray[np.float64] = 2 * (self.output - target)
        dW_out: NDArray[np.float64] = cost_delta @ self.hidden_state.T
        db_out: NDArray[np.float64] = np.sum(cost_delta, axis=1, keepdims=True)

        output_delta: NDArray[np.float64] = self.weight_output.T @ cost_delta

        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.biases)

        dc_next = np.zeros_like(self.cell_state)

        for t in reversed(self.cache):
            f, i, g, o = t["f_t"], t["i_t"], t["g_t"], t["o_t"]
            c_t, c_prev = t["c_t"], t["c_prev"]

            dc_t = dc_next + o * output_delta * derivative_tanh(c_t)

            dz_o = output_delta * tanh(c_t) * derivative_sigmoid(t["z_o"])
            dz_f = dc_t * c_prev * derivative_sigmoid(t["z_f"])
            dz_i = dc_t * g * derivative_sigmoid(t["z_i"])
            dz_g = dc_t * i * derivative_tanh(t["z_g"])

            dz = np.vstack([dz_f, dz_i, dz_g, dz_o])

            dW += dz @ t["concat"].T
            db += np.sum(dz, axis=1, keepdims=True)

            dconcat = self.weights.T @ dz
            dh_prev = dconcat[self.input_size :, :]
            dc_next = dc_t * f
            output_delta = dh_prev

        dW_out = clip_gradients(dW_out)
        db_out = clip_gradients(db_out)
        dW = clip_gradients(dW)
        db = clip_gradients(db)

        self.weight_output -= learning_rate * dW_out
        self.bias_output -= learning_rate * db_out
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

    def save(self, file_name: str = "model.pkl") -> None:
        dir = "./model"
        os.makedirs(dir, exist_ok=True)

        file_path = os.path.join(dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "input_size": self.input_size,
                    "state_size": self.state_size,
                    "batch": self.batch,
                    "weights": self.weights,
                    "biases": self.biases,
                    "weights_output": self.weight_output,
                    "bias_output": self.bias_output,
                },
                f,
            )

    @classmethod
    def load(cls, file_name: str) -> "LSTM_NeuralNetwork":
        dir = "./model"

        file_path = os.path.join(dir, file_name)
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        model = cls(
            input_size=data["input_size"],
            state_size=data["state_size"],
            batch=data["batch"],
        )

        model.weights = data["weights"]
        model.biases = data["biases"]
        model.weight_output = data["weights_output"]
        model.bias_output = data["bias_output"]

        return model

    def print(self) -> None:
        print("weights:")
        for weight in self.weights:
            print(weight)
        print()
        print("Biases:")
        for bias in self.biases:
            print(bias)
        print("Cell State:")
        print(self.cell_state)
        print("Hidden State:")
        print(self.hidden_state)
