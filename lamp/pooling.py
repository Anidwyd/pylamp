from .module import Module

import numpy as np


class MaxPool1D(Module):
    def __init__(self, k_size, stride=None):
        super().__init__()
        self.k_size = k_size
        self.stride = stride if stride else k_size

    def forward(self, X):
        batch, length, chan_in = X.shape
        d_out = (length - self.k_size) // self.stride + 1

        res = np.zeros((batch, d_out, chan_in))
        self.mask = np.zeros((batch, chan_in, length))

        for k in range(d_out):
            t1, t2 = self.stride * k, 2 * (self.k_size // 2) + k * self.stride + 1
            window = X[:, t1:t2, :]
            amax = np.amax(window, axis=1)
            res[:, k, :] = amax

            # saving max indices
            idx = np.where(X.transpose(0, 2, 1) == amax.reshape(batch, chan_in, 1))
            self.mask[idx] += 1

        self.mask.transpose(0, 2, 1)

        return res

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        batch, d_out, chan_in = delta.shape
        length = (d_out - 1) * self.stride + self.k_size
        res = np.zeros((batch, length, chan_in))

        for i, n in enumerate(self.indx):
            res[:, n, :] = delta[:, i, :]

        return res
