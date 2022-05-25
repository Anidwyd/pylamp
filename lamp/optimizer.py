import numpy as np


class Optimizer:
    def __init__(self, net, loss, eps=1e-3, early_stop=1e-3):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.early_stop = early_stop

    def step(self, batch_x, batch_y):
        yhat = self.net.forward(batch_x)
        loss = self.loss.forward(batch_y, yhat)

        delta = self.loss.backward(batch_y, yhat)
        self.net.backward(delta)

        self.net.update_parameters(self.eps)
        self.net.zero_grad()

        return yhat, np.mean(loss)

    def zero_grad(self):
        self.net.zero_grad()

    def SGD(self, datax, datay, batch_size, nb_iter=100):
        n = datax.shape[0]
        batch_size = min(batch_size, n)
        nb_batchs = n // batch_size

        inds = np.arange(n)
        np.random.shuffle(inds)

        train_losses = []

        i = 0
        for iteration in range(nb_iter):
            if i > nb_batchs:
                # shuffle images
                np.random.shuffle(inds)
                i = 0

            batch = inds[i * batch_size : (i + 1) * batch_size]
            yhat, loss = self.step(datax[batch], datay[batch])
            train_losses.append(loss)

            i += 1

            if not self.early_stop:
                continue

            if iteration > 100 and abs(train_losses[-2] - loss) < self.early_stop:
                print(f"early stopping activated")
                break

        return yhat, train_losses
