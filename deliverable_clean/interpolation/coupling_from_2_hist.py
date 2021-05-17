import numpy as np
import ot
import matplotlib.pyplot as plt
from cost_matrix import cost_matrix
import math


def coupling_from_2_hist(hist1 : np.ndarray, hist2 : np.ndarray, TEST_ALG) :
    """
    Simple, example, coupling matrix.
    Computes the coupling matrix of two images, turned into histograms.
    """
    #get the cost matrix for the images
    c, _ = cost_matrix(math.floor(np.sqrt(hist1.shape[0])), math.ceil(np.sqrt(hist1.shape[0])))

    coupling = TEST_ALG(hist1, hist2, c)

    np.save("coupling_matrixx.npy", coupling)

    return coupling