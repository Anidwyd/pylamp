# Pylamp - DIY Neural Network

The objective of this project is to implement a neural network. The implementation is inspired by the old versions of pytorch (in Lua, before the autograd) and similar implementations that allow to have very modular generic networks.

Each layer of the network is seen as a module and a network is thus made up of a set of modules. In particular, activation functions are also considered as modules.

![Modular architecture of a network](https://github.com/Anidwyd/pylamp/blob/main/docs/modular_net_arch.png?raw=true)