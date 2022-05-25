import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(yhat, datay, n_classes=10):
    C = np.zeros((n_classes, n_classes))

    for i in range(yhat.size):
        C[yhat[i], datay[i]] += 1

    C /= C.sum(axis=1, keepdims=1)

    return C


def compare_mc_perf(model, train_X, train_y, test_X, test_y):
    train_yhat = model.predict(train_X)
    train_score = round(model.score(train_X, train_y), 2)
    cm1 = confusion_matrix(train_yhat, train_y)

    test_yhat = model.predict(test_X)
    test_score = round(model.score(test_X, test_y), 2)
    cm2 = confusion_matrix(test_yhat, test_y)

    return [cm1, cm2], [train_score, test_score]


def plot_perf(cms, scores, figname, savedir="plots/", savefig=True):
    titles = [f"training score: {scores[0]}", f"testing score: {scores[1]}"]

    fig = plt.figure(constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=2, sharey=True)

    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm)
        ax.set_title(title)

    fig.colorbar(im, ax=axes, location="bottom")
    if savefig:
        plt.savefig(savedir + figname)
