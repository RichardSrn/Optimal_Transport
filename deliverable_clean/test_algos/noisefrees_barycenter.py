#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import ot


def noisefrees_bary():
    noisefrees = np.load("./data/noisefrees.npy")[:5]

    # a = []
    # for nf in noisefrees:
    #     a.append(nf.reshape(-1))
    # noisefrees = np.array(a).T
    # #
    # m = cost_matrix(50, 50) + 0.001
    # #
    # bary = ot.bregman.barycenter(A=noisefrees, M=m, reg=0.004)

    bary = ot.bregman.convolutional_barycenter2d(noisefrees, reg=0.004)

    np.save("./results/noisefrees_control/noisefrees_barycenter_" + str(noisefrees.shape[0]) + "_samples.npy", bary)

    # bary = np.load("./results/noisefrees_control/noisefrees_barycenter_2_samples.npy")

    bary = bary.reshape((50, 50))

    plt.figure(1, figsize=(10, 10))
    plt.imshow(bary)
    plt.title("barycenter")
    plt.savefig("./results/noisefrees_control/noisefrees_barycenter_" + str(noisefrees.shape[0]) + "_samples.png")
    plt.show()


if __name__ == "__main__":
    noisefrees_bary()
