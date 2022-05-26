from .module import Module

import numpy as np

from numba import njit


class MaxPool1D(Module):
    def __init__(self, k_size, stride=None):
        super().__init__()
        self.k_size = k_size
        self.stride = stride if stride else k_size

    @staticmethod
    @njit
    def forward_nb(X, k_size, stride):
        batch, length, chan_in = X.shape
        d_out = int(np.floor((length - k_size) / stride)) + 1

        output = np.zeros((batch, d_out, chan_in))
        saved_argmax = np.zeros_like(output, dtype=np.int32)

        idx_out = 0
        for i in range(0, length, stride):
            if idx_out == d_out:
                break
            for b in range(batch):
                for c in range(chan_in):
                    window = X[b, i : i + k_size, c]
                    output[b, idx_out, c] = np.max(window)
                    saved_argmax[b, idx_out, c] = np.argmax(window)
            idx_out += 1

        return output, saved_argmax

    def forward(self, X):
        # batch, length, chan_in = X.shape
        # self.d_out = (length - self.k_size) // self.stride + 1

        # res = np.zeros((batch, self.d_out, chan_in))

        # self.mask = np.zeros((batch, chan_in, length))

        # for k in range(self.d_out):
        #     t1, t2 = self.stride * k, 2 * (self.k_size // 2) + k * self.stride + 1
        #     window = X[:, t1:t2, :]
        #     amax = np.amax(window, axis=1)
        #     res[:, k, :] = amax

        #     # saving max indices
        #     idx = np.where(X.transpose(0, 2, 1) == amax.reshape(batch, chan_in, 1))
        #     self.mask[idx] += 1

        # self.mask = self.mask.transpose(0, 2, 1)

        # return res

        output, self.saved_argmax = self.forward_nb(X, self.k_size, self.stride)
        return output

    def backward_update_gradient(self, X, delta):
        pass

    @staticmethod
    @njit
    def backward_nb(delta, length, stride, saved_argmax):
        batch_, d_out, chan_in = delta.shape
        grad_input = np.zeros((batch_, length, chan_in))

        for b in range(batch_):
            for c in range(chan_in):
                for i in range(d_out):
                    grad_input[b, i * stride + saved_argmax[b, i, c], c] = delta[
                        b, i, c
                    ]
        return grad_input

    def backward_delta(self, X, delta):
        # size_diff = X.shape[1] - delta.shape[1]

        # d, r = size_diff // 2, size_diff % 2

        # delta = np.pad(
        #     delta, ((0, 0), (d, d + r), (0, 0)), "constant", constant_values=(0, 0)
        # )

        # return self.mask * delta

        length = X.shape[1]
        return self.backward_nb(delta, length, self.stride, self.saved_argmax)
