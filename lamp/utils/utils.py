import numpy as np


def add_bias(datax):
    return np.c_[datax, np.ones(datax.shape[0])]


def shuffle_batchs(n, batch_size, nb_batchs):
    inds = np.arange(n)
    np.random.shuffle(inds)
    batchs = [
        [j for j in inds[i * batch_size : (i + 1) * batch_size]]
        for i in range(nb_batchs)
    ]

    return batchs
