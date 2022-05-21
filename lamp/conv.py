from .module import Module

import numpy as np


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride=1):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self._parameters = (np.random.rand(k_size, chan_in, chan_out) * 2 - 1) * 1e-1
        self.zero_grad()

    def zero_grad(self):
        self._gradient = np.zeros((self.k_size, self.chan_in, self.chan_out))

    def forward(self, X):
        assert X.shape[2] == self.chan_in
        batch, length = X.shape[:2]
        d_out = (length - self.k_size) // self.stride + 1

        res = np.zeros((batch, d_out, self.chan_out))

        for k in range(d_out):
            t1, t2 = k * self.stride, 2 * (self.k_size // 2) + k * self.stride + 1
            res[:, k, :] = np.sum(
                X[:, t1:t2, :, np.newaxis] * self._parameters[np.newaxis, :, :, :],
                axis=(1, 2),
            )

        return res

    def backward_update_gradient(self, X, delta):
        raise NotImplementedError

    def backward_delta(self, X, delta):
        assert X.shape[2] == self.chan_in
        batch, length = X.shape[:2]
        d_out = (length - self.k_size) // self.stride + 1

        res = np.zeros((batch, d_out, self.chan_out))

        for k in range(d_out):
            t1, t2 = k * self.stride, 2 * (self.k_size // 2) + k * self.stride + 1
            res[:, k, :] = np.sum(
                np.flip(self._parameters, axis=1)[:, t1:t2, :, np.newaxis]
                * delta[np.newaxis, :, :, :],
                axis=(1, 2),
            )

        return res
