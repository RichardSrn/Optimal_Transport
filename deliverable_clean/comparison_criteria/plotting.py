import os

import matplotlib.pyplot as plt
import numpy as np


def test_same_scale(path="./results"):
    img1 = np.load(os.path.join(path, "entropic_reg_bary_convol/bary_lvl_0.000_mean_0.000_reg_0.4.npy"))
    img2 = np.load(os.path.join(path, "entropic_reg_bary_convol/bary_lvl_0.000_mean_0.000_reg_0.04.npy"))

    print(np.max(img1), np.min(img1))

    plt.figure(1, figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.title("img1")
    plt.imshow(img1)

    plt.subplot(2, 2, 2)
    plt.title("img2")
    plt.imshow(img2)

    plt.subplot(2, 2, 3)
    plt.title("img1 - [0-1] scale")
    plt.imshow(img1, vmin=0, vmax=1)

    plt.subplot(2, 2, 4)
    plt.title("img2 - [0-1] scale")
    plt.imshow(img2, vmin=0, vmax=1)

    plt.suptitle("plotting images with and without same scale.")
    plt.show()


if __name__ == "__main__":
    test_same_scale(path="../../deliverable_clean/test_algos/results")
