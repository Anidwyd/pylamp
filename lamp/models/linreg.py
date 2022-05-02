import numpy as np

from lamp import Linear, MSELoss


class LinearRegression:
    def fit(self, datax, datay, nb_iter=100, gradient_step=1e-3):
        assert datax.shape[0] == datay.shape[0], "X and Y have different batch sizes"
        _, input = datax.shape
        _, output = datay.shape

        self.loss = MSELoss()
        self.linear = Linear(input, output)

        self.loss_list = []

        for _ in range(nb_iter):
            # Forward
            yhat = self.linear.forward(datax)

            loss = self.loss.forward(datay, yhat)
            self.loss_list.append(np.mean(loss))

            # Backward
            delta = self.loss.backward(datay, yhat)
            delta = self.linear.backward_delta(datax, delta)

            self.linear.backward_update_gradient(datax, delta)

            # Update parameters
            self.linear.update_parameters(gradient_step)
            self.linear.zero_grad()

    def predict(self, datax):
        return self.linear.forward(datax)

    def score(self, datax, datay):
        return np.mean(self.predict(datax) == datay.reshape(-1, 1))
