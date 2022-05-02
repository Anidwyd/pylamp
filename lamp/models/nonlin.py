import numpy as np

from lamp import Sequential, Linear, MSELoss, Tanh, Sigmoid, Optimizer
from lamp.utils import add_bias


class NonLinear:
    def fit(self, datax, datay, hidden_size, nb_iter=100, gradient_step=1e-3):
        assert datax.shape[0] == datay.shape[0], "X and Y have different batch sizes"
        datax = add_bias(datax)

        _, input = datax.shape
        _, output = datay.shape

        self.loss = MSELoss()
        self.net = Sequential(
            Linear(input, hidden_size), Tanh(), Linear(hidden_size, output), Sigmoid()
        )

        self.loss_list = []
        self.optimizer = Optimizer(self.net, self.loss, eps=gradient_step)

        for _ in range(nb_iter):
            self.optimizer.step(datax, datay)

    def predict(self, datax):
        datax = add_bias(datax)
        yhat = self.net.forward(datax)
        return np.rint(yhat)

    def score(self, datax, datay):
        return np.mean(self.predict(datax) == datay)
