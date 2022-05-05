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
        batch, length = X.shape[:1]
        d_out = (length - self.k_size) // self.stride + 1

        out = np.zeros((batch, d_out, self.chan_out))

        for i in range(d_out):
            out[:, i, :] = np.sum(
                X[:, i * self.stride : self.k_size + i * self.stride + 1, :, np.newaxis]
                * self._parameters[np.newaxis, :, :, :],
                axis=(1, 2),
            )

        return out

    def backward_update_gradient(self, X, delta):
        raise NotImplementedError

    def backward_delta(self, X, delta):
        raise NotImplementedError
