import numpy as np

from lamp import Linear, MSELoss


class LinearRegression:
    def __init__(self, loss=MSELoss()):
        self.linear = None
        self.loss = loss
        self.loss_list = []

    def fit(self, datax, datay, nb_iter=100, gradient_step=1e-3):
        assert datax.shape[0] == datay.shape[0], "X and Y have different batch sizes"
        input, output = datax.shape[1], datay.shape[1]

        self.linear = Linear(input, output)

        while self.linear._parameters < 0:
            self.linear._parameters = 2 * (np.random.rand(input, output) - 0.5)

        for _ in range(nb_iter):
            yhat = self.linear.forward(datax)

            loss = self.loss.forward(datay, yhat)
            self.loss_list.append(np.mean(loss))

            delta = self.loss.backward(datay, yhat)
            delta = self.linear.backward_delta(datax, delta)

            self.linear.backward_update_gradient(datax, delta)

            self.linear.update_parameters(gradient_step)
            self.linear.zero_grad()

    def predict(self, datax):
        return self.linear.forward(datax)

    def score(self, datax, datay):
        return np.mean(self.predict(datax) == datay.reshape(-1, 1))
