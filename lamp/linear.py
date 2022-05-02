import numpy as np

from .module import Module


class Linear(Module):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        self._parameters = np.random.rand(self.input, self.output) * 2 - 1
        self.zero_grad()

    def zero_grad(self):
        self._gradient = np.zeros((self.input, self.output))

    def forward(self, X):
        assert X.shape[1] == self.input, f"{X.shape[1]} and {self.input} do not match"
        return X @ self._parameters

    def backward_update_gradient(self, X, delta):
        assert X.shape[1] == self.input, f"{X.shape[1]} and {self.input} do not match"
        self._gradient += X.T @ delta

    def backward_delta(self, X, delta):
        assert X.shape[1] == self.input, f"{X.shape[1]} and {self.input} do not match"
        return delta @ self._parameters.T
