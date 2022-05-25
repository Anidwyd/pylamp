import numpy as np


def add_bias(datax):
    return np.c_[datax, np.ones(datax.shape[0])]


def normalize_01(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def to_onehot(datay):
    onehot = np.zeros((datay.size, 10))
    onehot[np.arange(datay.size), datay] = 1
    return onehot


def noise_data(datax, noise_amount):
    xmax = datax.max()
    rand = np.random.rand(*datax.shape)
    return np.where(rand < noise_amount, np.random.rand(*datax.shape) * xmax, datax)
