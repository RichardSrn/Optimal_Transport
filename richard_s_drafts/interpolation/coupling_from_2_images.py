import numpy as np
import ot
import matplotlib.pyplot as plt


def coupling_from_2_images(hist1 : np.ndarray, hist2 : np.ndarray) :
    """
    Simple, example, coupling matrix.
    Computes the coupling matrix of two images, turned into histograms.
    """

    plt.imshow(hist1)
    plt.show()
    plt.imshow(hist2)
    plt.show()

