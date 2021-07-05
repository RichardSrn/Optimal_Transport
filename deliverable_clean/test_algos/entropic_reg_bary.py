#! /usr/bin/env python3
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import ot


def get_files():
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    onlyfiles = [file for file in onlyfiles if file[-5:] == "0.npy"]
    onlyfiles.sort()

    for file in onlyfiles:
        yield file


def cost_matrix(x_size, y_size, metric="sqeuclidean"):
    """Compute cost matrix which contains pairwise distances between locations of pixels"""
    nx, ny = x_size, y_size
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xv, yv = np.meshgrid(y, x)
    coors = np.vstack((xv.flatten(), yv.flatten())).T
    coors = coors[:, (1, 0)]
    C = ot.dist(coors, coors, metric=metric)
    return C


def hist_from_images(data):
    """
    Turns a 2D image into a 1D vector -a histogram-.
    """

    # turn the 2D images images into 1D histograms

    reshaped_data = []
    for i in range(data.shape[0]):
        img = data[i, :, :]
        hist = img.reshape(np.prod(img.shape))
        reshaped_data.append(hist)

    return np.array(reshaped_data)


def image_from_hist(hist: np.ndarray, size_x, size_y):
    image = hist.reshape((size_x, size_y))
    return image


def normalize(vectors):
    h = np.zeros(shape=vectors.shape)

    alpha = np.min(vectors)

    if np.isnan(alpha):
        alpha = 0

    for i in range(vectors.shape[0]):

        if not np.isnan(vectors[i, :]).any():
            sum_n_v = np.sum(vectors[i, :] - alpha)
            for n in range(vectors.shape[1]):
                h[i, n] = (vectors[i, n] - alpha) / sum_n_v
    return h


def entropic_reg_bary(reg=0.04, metric="sqeuclidean", size_x=50, size_y=50, plot=False, save=False, show=False):
    files = get_files()

    C = cost_matrix(size_x, size_y, metric=metric)  # metric = "cityblock"

    if plot:
        plt.figure(1, figsize=(15, 10))
        k = 1

    for file in files:
        title = "bary" + file[15:-4] + "_reg_" + str(reg)
        data = np.load("./data/" + file)
        data = normalize(data)
        data = hist_from_images(data).T
        bary = ot.bregman.barycenter_sinkhorn(data, C, reg=reg)
        bary = image_from_hist(bary, size_x, size_y)
        np.save("./results/entropic_reg_bary/" + title + ".npy", bary)

        if plot:
            plt.subplot(2, 3, k)
            plt.title(title[:len(title) // 2] + "\n" + title[len(title) // 2:])
            plt.imshow(bary)
            k += 1

    if show:
        plt.show()
    if save:
        plt.savefig(
            "./results/entropic_reg_bary/entropic_" + str(reg) + "_reg_" + str(data.shape[0]) + "_samples.png")


if __name__ == "__main__":
    reg = 0.4
    entropic_reg_bary(reg=reg)
