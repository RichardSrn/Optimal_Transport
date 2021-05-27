#! /usr/bin/env python3

from time import time

import matplotlib.pyplot as plt
import numpy as np


def plot_baryc(img1: np.ndarray, img2: np.ndarray, barycenter: np.ndarray, t=None, title="", show=False, save=False):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax1.imshow(img1)
    ax1.set_title("Image 1")

    ax2.imshow(img2)
    ax2.set_title("Image 2")

    ax3.imshow(barycenter)
    ax3.set_title("Barycenter")

    if t is not None:
        title = title + " - exec. : " + str(round(time() - t, 2)) + " s."

    plt.suptitle(title, fontsize=15)
    if show:
        print("\tShow the plot.")
        plt.show()
        print("\tDONE.")
    if save:
        print("\tSave the plot as image.")
        fig.savefig(title)
        print("\tDONE.")
