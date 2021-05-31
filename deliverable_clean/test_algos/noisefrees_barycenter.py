#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import ot


def noisefrees_bary():
    noisefrees = np.load("./data/noisefrees.npy")[:-1]

    # a = []
    # for nf in noisefrees:
    #     a.append(nf.reshape(-1))
    # noisefrees = np.array(a).T
    #
    # m = cost_matrix(50, 50)
    #
    # bary = ot.bregman.barycenter(A=noisefrees, M=m, reg=0.004)

    bary = ot.bregman.convolutional_barycenter2d(noisefrees, reg=0.004)

    np.save("./results/noisefrees_control/noisefrees_barycenter.npy", bary)

    # bary = np.load("./results/noisefrees_control/noisefrees_barycenter.npy")

    bary = bary.reshape((50, 50))

    plt.figure(1, figsize=(15, 15))
    for i in range(4):
        for j in range(4):
            if 4 * i + j >= noisefrees.shape[0]:
                break
            plt.subplot(4, 4, 4 * i + j + 1)
            plt.imshow(noisefrees[4 * i + j])
    plt.subplot(3, 3, 9)
    plt.imshow(bary)
    plt.title("barycenter")
    plt.show()
    plt.savefig("./results/noisefrees_control/noisefrees_barycenter.png")


if __name__ == "__main__":
    noisefrees_bary()
