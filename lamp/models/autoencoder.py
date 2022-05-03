import numpy as np

from lamp import Sequential, Linear, BCELoss, Tanh, Sigmoid, Optimizer


class AutoEncoder:
    def __init__(self, loss=BCELoss()):
        self.loss = loss
        self.loss_list = []

    def fit(self, datax, datay, hidden, nb_iter=100, gradient_step=1e-3, batch_size=0):
        assert datax.shape[0] == datay.shape[0], "X and Y have different batch sizes"
        input, output = datax.shape[1], datay.shape[1]

        encoder = [Linear(input, hidden), Tanh(), Linear(hidden, output), Tanh()]
        decoder = [Linear(output, hidden), Tanh(), Linear(hidden, input), Sigmoid()]

        self.net = Sequential(*encoder, *decoder)
        self.optimizer = Optimizer(self.net, self.loss, eps=gradient_step)

        if batch_size > 0:
            self.loss_list = self.optimizer.SGD(datax, datax, batch_size, nb_iter)[0]
            return

        for _ in range(nb_iter):
            loss = self.optimizer.step(datax, datax)[0]
            self.loss_list.append(np.mean(loss))

    def predict(self, datax):
        return self.net.forward(datax)

    def score(self, datax, datay):
        return np.mean(self.predict(datax) == datay)
