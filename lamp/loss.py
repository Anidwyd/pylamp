import numpy as np

from .activation import Softmax


class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y, yhat):
        assert y.shape == yhat.shape, f"{y.shape} and {yhat.shape} do not match"
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, f"{y.shape} and {yhat.shape} do not match"
        return -2 * (y - yhat)


class CELoss(Loss):
    def forward(self, y, yhat):
        return -np.sum(y * yhat, axis=1)

    def backward(self, y, yhat):
        return -y


class SMCELoss(Loss):
    def forward(self, y, yhat):
        return -np.sum(y * yhat, axis=1) + np.log(np.sum(np.exp(yhat), axis=1))

    def backward(self, y, yhat):
        s = Softmax().forward(yhat)
        return -y + s * (1 - s)


class BCELoss(Loss):
    def forward(self, y, yhat):
        return -(
            y * np.maximum(-100, np.log(yhat + 0.01))
            + (1 - y) * np.maximum(-100, np.log(1 - yhat + 0.01))
        )

    def backward(self, y, yhat):
        return -((y / (yhat + 1e-2)) - ((1 - y) / (1 - yhat + 1e-2)))
