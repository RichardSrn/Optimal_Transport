#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import ot
import pylab as pl


def interpolation(mu_s=None, mu_t=None):
    # set parameters
    rng = np.random.RandomState(4269)

    if type(mu_s) == None:
        # Generate the distributions
        indexes = rng.randint(0, 199, size=2)
        mu_s, mu_t = np.load("../PRNI2018_TLp_bary/artificial_data_nn.npy")[indexes]
        mu_s /= mu_s.sum()
        mu_t /= mu_t.sum()

    if mu_s.shape != mu_t.shape:
        print("Error, images are not of the same shape.")
        print("\tmu_s.shape is ", mu_s.shape)
        print("\tmu_t.shape is ", mu_t.shape)
        return 0

    x_size, y_size = mu_s.shape

    grid = []
    for i in range(mu_s.shape[0]):
        for j in range(mu_s.shape[1]):
            grid.append([i, j])
    grid = np.array(grid)

    dist1 = np.hstack(mu_s)
    dist2 = np.hstack(mu_t)

    plot1 = False
    if plot1:
        plt.figure(1, figsize=(5, 5))
        plt.clf()
        plt.xlim(-1, x_size)
        plt.ylim(-1, y_size)
        for i in range(dist1.shape[0]):
            if dist1[i] > 0:
                plt.scatter(grid[i, 0], grid[i, 1], color="blue", alpha=dist1[i] * dist1.max(), label="mu_s")
            if dist2[i] > 0:
                plt.scatter(grid[i, 0], grid[i, 1], color="red", alpha=dist2[i] * dist2.max(), label="mu_t")

    m = ot.dist(grid, grid, metric="euclidean")
    ot_emd = ot.emd(dist1, dist2, M=m / m.max())
    print('\not_emd :\n', ot_emd)

    plot2 = True
    if plot2:
        plt.figure(2, figsize=(5, 5))
        plt.clf()
        plt.xlim(-1, x_size)
        plt.ylim(-1, y_size)
        for i in range(len(grid)):
            if i % 100 == 0:
                print(i, "/", len(grid), " -- work in progress...")
            for j in range(len(grid)):
                if dist1[i] > 0 and dist2[j] > 0:
                    x = [grid[i, 0], grid[j, 0]]
                    y = [grid[i, 1], grid[j, 1]]
                    pl.plot(x, y, '-k', alpha=ot_emd[i, j] / ot_emd.max())
        for i in range(dist1.shape[0]):
            if dist1[i] > 0:
                plt.scatter(grid[i, 0], grid[i, 1], color="blue", alpha=dist1[i] * dist1.max(), label="mu_s")
            if dist2[i] > 0:
                plt.scatter(grid[i, 0], grid[i, 1], color="red", alpha=dist2[i] * dist2.max(), label="mu_t")

    dists = np.array([mu_s,mu_t])
    print(dists.shape)
    baryc = ot.bregman.convolutional_barycenter2d(dists, reg=0.0001)
    print('\nbaryc :\n', baryc)

    plot3 = False
    if plot3:
        plt.figure(3, figsize=(5, 5))
        plt.clf()
        plt.xlim(-1, x_size)
        plt.ylim(-1, y_size)
        for i in range(dist1.shape[0]):
            if dist1[i] > 0:
                plt.scatter(grid[i, 0], grid[i, 1], color="blue", alpha=dist1[i] * dist1.max(), label="mu_s")
            if dist2[i] > 0:
                plt.scatter(grid[i, 0], grid[i, 1], color="red", alpha=dist2[i] * dist2.max(), label="mu_t")
            if baryc:
                pass

    plt.show()


if __name__ == '__main__':
    mu_s = np.array([[0, .5, .5],
                     [.5, 0, .5],
                     [0, 0, 0]])
    mu_t = np.array([[.5, 0, 0],
                     [0, .5, 0],
                     [.5, 0, .5]])
    interpolation(mu_s, mu_t)
