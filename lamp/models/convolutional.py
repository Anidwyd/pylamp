import numpy as np

from lamp import (
    Sequential,
    Conv1D,
    MaxPool1D,
    Flatten,
    Linear,
    ReLU,
    Optimizer,
    SMCELoss,
)
from lamp.utils import add_bias


class Convolutional:
    def __init__(self, loss=SMCELoss()):
        self.loss = loss
        self.train_losses = []
        self.valid_losses = []

    def fit(
        self,
        datax,
        datay,
        hidden=None,
        nb_iter=100,
        gradient_step=1e-2,
        batch_size=0,
        early_stop=1e-3,
    ):
        assert datax.shape[0] == datay.shape[0], "X and Y have different batch sizes"

        self.net = Sequential(
            Conv1D(3, 1, 32),
            MaxPool1D(2, 2),
            Flatten(),
            Linear(4064, 100),
            ReLU(),
            Linear(100, 10),
        )
        self.optimizer = Optimizer(
            self.net, self.loss, eps=gradient_step, early_stop=early_stop
        )

        if batch_size > 0:
            self.train_losses = self.optimizer.SGD(datax, datay, batch_size, nb_iter)[1]
            return

        for _ in range(nb_iter):
            loss = self.optimizer.step(datax, datay)[1]
            self.train_losses.append(np.mean(loss))

    def predict(self, datax):
        yhat = self.net.forward(datax)
        return np.argmax(yhat, axis=1)

    def score(self, datax, datay):
        return np.mean(self.predict(datax) == datay)
