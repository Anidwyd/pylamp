import numpy as np


class Optimizer:
    def __init__(self, net, loss, eps):
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

        return loss

    def SGD(self, net, batch_x, batch_size, nb_iter):
        pass
