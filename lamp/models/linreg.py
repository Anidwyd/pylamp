import numpy as np

from lamp import Linear, MSELoss


class LinearRegression:
    def __init__(self, loss=MSELoss()):
        self.loss = loss
        self.loss_list = []

    def fit(self, datax, datay, nb_iter=100, gradient_step=1e-3):
        assert datax.shape[0] == datay.shape[0], "X and Y have different batch sizes"
        input, output = datax.shape[1], datay.shape[1]

        self.net = Linear(input, output)

        for _ in range(nb_iter):
            # Forward
            yhat = self.net.forward(datax)
            loss = self.loss.forward(datay, yhat)
            self.loss_list.append(np.mean(loss))

            # Backward
            delta = self.loss.backward(datay, yhat)
            delta = self.net.backward_delta(datax, delta)
            self.net.backward_update_gradient(datax, delta)

            # Update parameters
            self.net.update_parameters(gradient_step)
            self.net.zero_grad()

    def predict(self, datax):
        return self.net.forward(datax)

    def score(self, datax, datay):
        return np.mean(self.predict(datax) == datay.reshape(-1, 1))
