#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def plot_baryc(img1 : np.ndarray, img2 : np.ndarray, barycenter : np.ndarray, title):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,5))
    ax1.imshow(img1)
    ax1.set_title("Image 1")

    ax2.imshow(img2)
    ax2.set_title("Image 2")

    ax3.imshow(barycenter)
    ax3.set_title("Barycenter")

    plt.suptitle(title, fontsize=15)
    plt.show()