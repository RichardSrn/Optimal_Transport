import numpy as np

from cost_matrix import cost_matrix


def coupling_from_2_hist(hist1: np.ndarray, hist2: np.ndarray, TEST_ALG, size_x, size_y):
    """
    Simple, example, coupling matrix.
    Computes the coupling matrix of two images, turned into histograms.
    """

    # get the cost matrix for the images
    # c, _ = cost_matrix(math.floor(np.sqrt(hist1.shape[0])), math.ceil(np.sqrt(hist1.shape[0])))
    c, _ = cost_matrix(size_x, size_y)

    # get the coupling matrix from the algorithm to test.
    coupling = TEST_ALG(hist1, hist2, c)

    return coupling
