import numpy as np

from lamp.activation import Softmax


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


class CELoss(Loss):
    def forward(self, y, yhat):
        return -np.sum(y * yhat, axis=1)

    def backward(self, y, yhat):
        return -y


class SMCELoss(Loss):
    def forward(self, y, yhat):
        return -np.sum(y * yhat, axis=1) + np.log(np.exp(yhat), axis=1)

    def backward(self, y, yhat):
        s = Softmax().forward(y)
        return -y + s * (1 - s)
