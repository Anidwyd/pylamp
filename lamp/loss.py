from turtle import forward
import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y, yhat):
        assert y.shape == yhat.shape, f"Shapes {y.shape} and {yhat.shape} do not match"
        return (np.linalg.norm(y - yhat, axis=1) ** 2).reshape(-1, 1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, f"Shapes {y.shape} and {yhat.shape} do not match"
        return -2 * (y - yhat)


class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError


class CE(Loss):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError
