import numpy as np

from .module import Module


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        return (1 - np.tanh(X) ** 2) * delta


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        return (np.exp(-X) / (1 + np.exp(-X)) ** 2) * delta


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        exp = np.exp(X)
        return exp / exp.sum(axis=1).reshape(-1, 1)

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        s = self.forward(X)
        return s * (1 - s) * delta
