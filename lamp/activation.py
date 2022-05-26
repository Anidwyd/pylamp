import numpy as np
from torch import exp2_

from .module import Module


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.tanhX = np.tanh(X)
        return self.tanhX

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        return (1 - self.tanhX ** 2) * delta


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.sigmoid = 1 / (1 + np.exp(-X))
        return self.sigmoid

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        return self.sigmoid * (1 - self.sigmoid) * delta


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        input = X - np.max(X, axis=1, keepdims=True)
        exp_ = np.exp(input)
        self.soft = exp_ / exp_.sum(axis=1).reshape(-1, 1)
        return self.soft

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        return self.soft * (1 - self.soft) * delta


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X * (X > 0)

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        return 1.0 * (X > 0) * delta
