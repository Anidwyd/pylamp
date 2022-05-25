import numpy as np

from lamp import Sequential, Linear, BCELoss, Tanh, Sigmoid, Optimizer


class AutoEncoder:
    def __init__(self, loss=BCELoss()):
        self.loss = loss
        self.train_losses = []
        self.valid_losses = []

    def fit(
        self,
        datax,
        datay,
        hidden,
        latent,
        nb_iter=100,
        gradient_step=1e-4,
        batch_size=0,
        early_stop=1e-6,
    ):
        input = datax.shape[1]

        encoder = []
        decoder = []
        layers = [input] + list(hidden) + [latent]

        for j in range(len(layers) - 1):
            encoder += [Linear(layers[j], layers[j + 1]), Tanh()]
            decoder += [Linear(layers[-j - 1], layers[-j - 2]), Tanh()]

        decoder[-1] = Sigmoid()

        self.net = Sequential(*encoder, *decoder)
        self.optimizer = Optimizer(
            self.net, self.loss, eps=gradient_step, early_stop=early_stop
        )

        if batch_size > 0:
            self.train_losses = self.optimizer.SGD(datax, datay, batch_size, nb_iter)[1]
            return

        for epoch in range(nb_iter):
            loss = np.mean(self.optimizer.step(datax, datax)[0])
            self.train_losses.append(loss)

            if len(self.train_losses) > 10 and abs(self.train_losses[-2] - loss) < 1e-6:
                print(f"early stopping at epoch {epoch}")
                return

    def predict(self, datax):
        return self.net.forward(datax)

    def score(self, datax, datay):
        return np.mean(self.predict(datax) == datay)
