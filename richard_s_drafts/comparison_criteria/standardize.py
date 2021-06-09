import numpy as np


def standardize(vectors):
    h = np.zeros(shape=vectors.shape)

    alpha = np.min(vectors)

    for i in range(vectors.shape[0]) :

        sum_n_v = np.sum(vectors[i,:] - alpha)

        for n in range(vectors.shape[1]) :
            h[i,n] = ( vectors[i,n] - alpha ) / sum_n_v

    return h
