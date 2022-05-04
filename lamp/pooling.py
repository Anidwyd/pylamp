from .module import Module


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        raise NotImplementedError

    def backward_update_gradient(self, X, delta):
        pass

    def backward_delta(self, X, delta):
        raise NotImplementedError
