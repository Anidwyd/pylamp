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

        # for k in range(d_out):
        #     t1, t2 = self.stride * k, 2 * (self.k_size // 2) + k * self.stride + 1
        #     res[:, k, :] = np.amax(X[:, t1:t2, :], axis=1)

        self.indx = []

        for k in range(d_out):
            t1, t2 = self.stride * k, 2 * (self.k_size // 2) + k * self.stride + 1
            tupleA = np.argmax(X[:, t1:t2, :], axis=1)
            print("tupl",tupleA[0][0])
           
            listProv = [0] * (tupleA[0][0].item()-1) + tupleA[0].tolist()# renvois pas 1 mais indice a corriger
            print(listProv)
            self.indx += listProv
            res[:, k, :] = X[:,tupleA[0], :].squeeze(axis = 1)
            print(self.indx)
        self.indx = np.array(self.indx)
        return res
        

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        batch, d_out, chan_in = delta.shape
        length = (d_out - 1) * self.stride + self.k_size

        res = np.zeros((batch, length, chan_in))
        
        res  = delta * self.indx[np.newaxis,:,np.newaxis]
        return res
