from .module import Module


class Sequential(Module):
    def __init__(self, *args: Module):
        super().__init__()
        self._modules = args
        self._inputs = []

    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()

    def forward(self, input):
        self._inputs = []
        for module in self._modules:
            input = module.forward(input)
            self._inputs.append(input)
        return input

    def update_parameters(self, gradient_step=1e-3):
        for module in self._modules:
            module.update_parameters(gradient_step)

    def backward(self, delta):
        for idx in range(len(self._modules) - 1, 0, -1):
            self._modules[idx].backward_update_gradient(self._inputs[idx - 1], delta)
            delta = self._modules[idx].backward_delta(self._inputs[idx - 1], delta)
