import numpy as np
import matplotlib.pyplot as plt

from .utils import normalize_01
from keras.datasets import mnist, fashion_mnist


def load_mnist(dataset="mnist"):
    data = mnist if dataset == "mnist" else fashion_mnist

    (train_X, train_y), (test_X, test_y) = data.load_data()
    train_batch, test_batch = train_X.shape[0], test_X.shape[0]

    train_X = normalize_01(train_X.reshape(train_batch, -1))
    test_X = normalize_01(test_X.reshape(test_batch, -1))

    return (train_X, train_y), (test_X, test_y)


def get_mnist(l, datax, datay):
    if type(l) != list:
        resx = datax[datay == l, :]
        resy = datay[datay == l]
        return resx, resy

    tmp = list(zip(*[get_mnist(i, datax, datay) for i in l]))
    tmpx, tmpy = np.vstack(tmp[0]), np.hstack(tmp[1])
    return tmpx, tmpy


def show_mnist(data, cmap="gray"):
    plt.imshow(data.reshape(28, 28), interpolation="nearest", cmap=cmap)
