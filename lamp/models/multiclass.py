import numpy as np

from lamp import Sequential, Linear, SMCELoss, Tanh, Sigmoid, Optimizer
from lamp.utils import add_bias


class MultiClass:
    def __init__(self, loss=SMCELoss()):
        self.loss = loss
        self.loss_list = []

    def fit(self, datax, datay, hidden, nb_iter=100, gradient_step=1e-3, batch_size=0):
        datax = add_bias(datax)

        assert datax.shape[0] == datay.shape[0], "X and Y have different batch sizes"
        input, output = datax.shape[1], datay.shape[1]

        self.net = Sequential(
            Linear(input, hidden), Tanh(), Linear(hidden, output), Sigmoid()
        )

        self.optimizer = Optimizer(self.net, self.loss, eps=gradient_step)

        if batch_size > 0:
            self.loss_list = self.optimizer.SGD(datax, datay, batch_size, nb_iter)[0]
            return

        for _ in range(nb_iter):
            loss = self.optimizer.step(datax, datay)[0]
            self.loss_list.append(np.mean(loss))

    def predict(self, datax):
        datax = add_bias(datax)
        yhat = self.net.forward(datax)
        return np.argmax(yhat, axis=1)

    def score(self, datax, datay):
        return np.mean(self.predict(datax) == datay)
