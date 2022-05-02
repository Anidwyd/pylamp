import numpy as np

from lamp.utils import add_bias, shuffle_batchs


class Optimizer:
    def __init__(self, net, loss, eps=1e-3):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        yhat = self.net.forward(batch_x)
        loss = self.loss.forward(batch_y, yhat)
        delta = self.loss.backward(batch_y, yhat)
        self.net.backward(delta)

        self.net.update_parameters(self.eps)
        self.net.zero_grad()

        return yhat, loss

    def SGD(self, datax, datay, batch_size, nb_iter=100):
        datax = add_bias(datax)

        n = datax.shape[0]
        nb_batchs = n // batch_size

        inds = np.arange(n)
        np.random.shuffle(inds)

        list_loss = []

        i = 0
        for _ in range(nb_iter):

            if i > nb_batchs:
                np.random.shuffle(inds)
                i = 0

            batch = inds[i * batch_size : (i + 1) * batch_size]
            yhat, loss = self.step(datax[batch], datay[batch])
            list_loss.append(loss)

            i += 1

        return yhat, list_loss
