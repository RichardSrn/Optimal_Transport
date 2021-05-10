#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import ot
import pylab as pl


def interpolation(mu_s=None, mu_t=None):
    # set parameters
    rng = np.random.RandomState(4269)

    if mu_s == None:
        # Generate the distributions
        indexes = rng.randint(0, 199, size=2)
        mu_s, mu_t = np.load("../PRNI2018_TLp_bary/artificial_data_nn.npy")[indexes]
        mu_s /= mu_s.sum()
        mu_t /= mu_t.sum()

    mu_s_pos = []
    mu_t_pos = []
    for i in range(mu_s.shape[0]):
        for j in range(mu_s.shape[1]):
            mu_s_pos.append([i, j])
            mu_t_pos.append([i, j])
    mu_s_pos = np.array(mu_s_pos)
    mu_t_pos = np.array(mu_t_pos)

    mu_s = np.hstack(mu_s)
    mu_t = np.hstack(mu_t)

    plot1 = False
    if plot1:
        plt.figure(1, figsize=(5, 5))
        plt.clf()
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        for i in range(mu_s.shape[0]):
            if mu_s[i] > 0:
                plt.scatter(mu_s_pos[i, 0], mu_s_pos[i, 1], color="blue", alpha=mu_s[i] * 10)
            if mu_t[i] > 0:
                plt.scatter(mu_t_pos[i, 0], mu_t_pos[i, 1], color="red", alpha=mu_t[i] * 10)
        plt.show()

    m = ot.dist(mu_s_pos, mu_t_pos, metric="euclidean")
    ot_emd = ot.emd(mu_s, mu_t, M=m / m.max())

    plot2 = True
    if plot2:
        plt.figure(2, figsize=(5, 5))
        plt.clf()
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        for i in range(len(mu_s_pos)):
            if i % 50 == 0:
                print(i, "/", len(mu_s_pos))
            for j in range(len(mu_t_pos)):
                if mu_s[i] > 0 and mu_t[j] > 0:
                    x = [mu_s_pos[i, 0], mu_t_pos[j, 0]]
                    y = [mu_s_pos[i, 1], mu_t_pos[j, 1]]
                    pl.plot(x, y, '-k', alpha=ot_emd[i, j] / ot_emd.max())
        for i in range(mu_s.shape[0]):
            if mu_s[i] > 0:
                plt.scatter(mu_s_pos[i, 0], mu_s_pos[i, 1], color="blue", alpha=mu_s[i] * 10)
            if mu_t[i] > 0:
                plt.scatter(mu_t_pos[i, 0], mu_t_pos[i, 1], color="red", alpha=mu_t[i] * 10)
        plt.show()


if __name__ == '__main__':
    interpolation()
