from .module import Module

import numpy as np


class MaxPool1D(Module):
    def __init__(self, k_size, stride=None):
        super().__init__()
        self.k_size = k_size
        self.stride = stride if stride else k_size
        self.indx = list()

    def forward(self, X):
        batch, length, chan_in = X.shape
        d_out = (length - self.k_size) // self.stride + 1

        res = np.zeros((batch, d_out, chan_in))

        # for k in range(d_out):
        #     t1, t2 = self.stride * k, 2 * (self.k_size // 2) + k * self.stride + 1
        #     res[:, k, :] = np.amax(X[:, t1:t2, :], axis=1)
        
        for k in range(d_out):
            t1, t2 = self.stride * k, 2 * (self.k_size // 2) + k * self.stride + 1
            self.indx = np.where(X == np.amax(X[:, t1:t2, :], axis=1))
            res[:, k, :] = X[:,self.indx:,]

        return res

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        batch,  d_out, chan_in = delta.shape
        length = (d_out - 1) * self.stride + self.k_size
        
        res = np.zeros((batch, length, chan_in))

        res[np.repeat(range(batch),chan_in),self.indx,list(range(chan_in))*batch]
        


        return res
