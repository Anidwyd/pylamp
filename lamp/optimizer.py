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

    def SGD(self, datax, datay, batch_size, nb_iter):
        for _ in range(100):
            yhat, loss = self.step(datax, datay)
