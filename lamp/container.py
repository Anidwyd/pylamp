from itertools import islice
from collections import OrderedDict
from typing import Union
import operator

from .module import Module


class Sequential(Module):
    def __init__(self, *args: Module):
        super().__init__()
        self._modules = dict()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._modules[key] = module
        else:
            for idx, module in enumerate(args):
                self._modules[str(idx)] = module

        self.inputs = []

    def _get_item_by_index(self, iterator, idx):
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_index(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module):
        key = self._get_item_by_index(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module: Module):
        self._modules[str(len(self))] = module
        return self

    def forward(self, input):
        self.inputs = []
        for module in self:
            input = module.forward(input)
            self.inputs.append(input)
        return input

    def backward(self, delta):
        for idx in range(len(self) - 1, 0, -1):
            self[idx].backward_update_gradient(self.inputs[idx - 1], delta)
            delta = self[idx].backward_delta(self.inputs[idx - 1], delta)
