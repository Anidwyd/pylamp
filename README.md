# Pylamp - DIY Neural Network

The objective of this project is to implement a neural network. The implementation is
inspired by the old versions of pytorch (in Lua, before the autograd) and similar
implementations that allow to have very modular generic networks.  
Each layer of the network is seen as a module and a network is thus made up of a set of
modules. In particular, activation functions are also considered as modules.

![Modular architecture of a network](https://github.com/Anidwyd/pylamp/blob/main/docs/modular_net_arch.png?raw=true)

## Installation

Clone the repo then :

```
pip install -e .
```

We also recommand installing the following librairies :
[my_gym](https://github.com/osigaud/my_gym),
[rllab](https://github.com/rll/rllab),
[pybox2D](https://github.com/pybox2d/pybox2d)
and osigaud's fork of [salina](https://github.com/osigaud/salina).