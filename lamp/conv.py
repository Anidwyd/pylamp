from .module import Module

import numpy as np

from numba import njit


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

    @staticmethod
    @njit
    def forward_nb(X, params, stride):
        chan_out, k_size, _ = params.shape
        batch, length, _ = X.shape
        d_out = (length - k_size) // stride + 1
        output = np.zeros((batch, d_out, chan_out))
        idx_out = 0

        for i in range(0, length, stride):
            if idx_out == d_out:
                break

            for b in range(batch):
                window = X[b, i : i + k_size, :]
                for c in range(chan_out):
                    kernel = params[c, :, :]
                    output[b, idx_out, c] = np.sum(kernel * window)

            idx_out += 1

        return output

    def forward(self, X):
        # assert X.shape[2] == self.chan_in

        # batch, length = X.shape[:2]
        # self.d_out = (length - self.k_size) // self.stride + 1

        # res = np.zeros((batch, self.d_out, self.chan_out))

        # for k in range(self.d_out):
        #     t1, t2 = k * self.stride, 2 * (self.k_size // 2) + k * self.stride + 1
        #     window = X[:, t1:t2, :, np.newaxis]
        #     res[:, k, :] = np.sum(
        #         window * self._parameters[np.newaxis, :, :, :], axis=(1, 2)
        #     )

        # return res
        # batch, length = X.shape[:2]
        return self.forward_nb(X, self._parameters, self.stride)

    def backward_update_gradient(self, X, delta):
        res = np.zeros((self.k_size, self.chan_in, self.chan_out))

        for k in range(self.k_size):
            t1, t2 = k * self.stride, self.d_out + k * self.stride
            window = X[:, t1:t2, :, np.newaxis]
            res[k, :, :] = np.sum(window * delta[:, :, np.newaxis, :], axis=(0, 1))

        return res

    @staticmethod
    @njit
    def backward_nb(X, params, delta, stride):
        chan_out, k_size, _ = params.shape
        batch, length, _ = X.shape
        d_out = delta.shape[1]

        grad_input = np.zeros_like(X)
        grad_W = np.zeros_like(params)
        idx_out = 0
        for i in range(0, length, stride):
            if idx_out == d_out:
                break

            for c in range(chan_out):
                for b in range(batch):
                    kernel = params[c, :, :]  # ksz, chan_in
                    grad_input[b, i : i + k_size, :] += kernel * delta[b, idx_out, c]
                    grad_W[c, :, :] += X[b, i : i + k_size, :] * delta[b, idx_out, c]

            idx_out += 1

        return grad_input, grad_W / batch

    def backward_delta(self, X, delta):
        # assert X.shape[2] == self.chan_in
        # batch = X.shape[0]

        # res = np.zeros((batch, self.d_out, self.chan_in))

        # for k in range(self.d_out):
        #     t1, t2 = 0, self.k_size
        #     d1 = -k * self.stride - 2 + self.k_size
        #     d2 = -(k * self.stride + 2)

        #     if k == 0:
        #         t1 = 0
        #         t2 = k + 1
        #         d1 = None
        #         d2 = -1

        #     elif k <= self.k_size - 1:
        #         t1 = 0
        #         t2 = k + 1
        #         d1 = -1
        #         d2 = -k - 2

        #     elif k > self.d_out - self.k_size - 1:
        #         t1 = self.k_size - self.d_out + k
        #         t2 = self.k_size
        #         d1 = -k + 2
        #         d2 = 0

        #     res[:, k, :] = np.sum(
        #         np.flip(self._parameters, axis=1)[np.newaxis, t1:t2, :, :]
        #         * delta[:, d2:d1, np.newaxis, :],
        #         axis=(1, 3),
        #     )

        # return res
        # output,
        return self.backward_nb(X, self._parameters, delta, self.stride)
