from .module import Module


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        assert len(X.shape) == 3
        self.batch, self.length, self.chan_in = X.shape
        return X.reshape(self.batch, -1)

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        assert len(delta.shape) == 2
        return delta.reshape(self.batch, self.length, self.chan_in)
