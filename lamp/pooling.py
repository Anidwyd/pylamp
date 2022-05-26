from .module import Module

import numpy as np


class MaxPool1D(Module):
    def __init__(self, k_size, stride=None):
        super().__init__()
        self.k_size = k_size
        self.stride = stride if stride else k_size

    def forward(self, X):
        batch, length, chan_in = X.shape
        self.d_out = (length - self.k_size) // self.stride + 1

        res = np.zeros((batch, self.d_out, chan_in))

        self.mask = np.zeros((batch, chan_in, length))

        for k in range(self.d_out):
            t1, t2 = self.stride * k, 2 * (self.k_size // 2) + k * self.stride + 1
            window = X[:, t1:t2, :]
            amax = np.amax(window, axis=1)
            res[:, k, :] = amax

            # saving max indices
            idx = np.where(X.transpose(0, 2, 1) == amax.reshape(batch, chan_in, 1))
            self.mask[idx] += 1

        self.mask = self.mask.transpose(0, 2, 1)

        return res

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        size_diff = X.shape[1] - delta.shape[1]

        d, r = size_diff // 2, size_diff % 2

        delta = np.pad(
            delta, ((0, 0), (d, d + r), (0, 0)), "constant", constant_values=(0, 0)
        )

        return self.mask * delta
