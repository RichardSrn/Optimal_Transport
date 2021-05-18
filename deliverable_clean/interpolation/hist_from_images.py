import numpy as np


def hist_from_images(img1: np.ndarray, img2: np.ndarray):
    """
    Turns a 2D image into a 1D vector -a histogram-.
    """

    # turn the 2D images images into 1D histograms
    hist1 = img1.reshape((np.prod(img1.shape, -1)))
    hist2 = img2.reshape((np.prod(img2.shape, -1)))

    return (hist1, hist2)
