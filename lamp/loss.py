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
        return np.log(np.sum(np.exp(yhat), axis=1)) - np.sum(y * yhat, axis=1)

    def backward(self, y, yhat):
        exp_ = np.exp(yhat)
        soft = exp_ / exp_.sum(axis=1).reshape(-1, 1)
        return -y + soft * (1 - soft)


class BCELoss(Loss):
    # def forward(self, y, yhat):
    #     return (y - 1) * np.log(1 - yhat + 1e-100) - y * np.log(yhat + 1e-100)

    # def backward(self, y, yhat):
    #     return (yhat - y) / (yhat * (1 - yhat) + 1e-100)

    def forward(self, y, yhat):
        term_0 = (y - 1) * np.maximum(-100, np.log(1 - yhat + 1e-100))
        term_1 = y * np.maximum(-100, np.log(yhat + 1e-100))
        return term_0 - term_1

    def backward(self, y, yhat):
        return (yhat - y) / np.maximum(yhat * (1 - yhat), 1e-12)

    # def forward(self, y, yhat):
    #     return -(
    #         y * np.maximum(np.log(np.maximum(yhat, 1e-10)), -100)
    #         - (1 - y) * np.maximum(np.log(np.maximum(1 - yhat, 1e-10)), -100)
    #     )

    # def backward(self, y, yhat):

    #     R1 = -y * (1 / (yhat + 1e-8))
    #     R2 = (1 - y) * (1 / (1 - (yhat)))
    #     R = R1 + R2
    #     return 1 / yhat.shape[1] * R
