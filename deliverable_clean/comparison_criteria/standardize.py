import os

import numpy as np


def get_files(path="./data"):
    directories = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    directories.remove('noisefrees_control')

    tree = {d: [] for d in directories}

    for d in directories:
        tree[d] = [f for f in os.listdir(os.path.join(path, d)) if os.path.isfile(os.path.join(path, d, f))]
        tree[d] = [file for file in tree[d] if file[-4:] == ".npy"]
        tree[d].sort()

    return tree


def standardize(vectors):
    h = np.zeros(shape=vectors.shape)

    alpha = np.min(vectors)

    if np.isnan(alpha):
        alpha = 0

    for i in range(vectors.shape[0]):

        if not np.isnan(vectors[i, :]).any():
            sum_n_v = np.sum(vectors[i, :] - alpha)
            for n in range(vectors.shape[1]):
                h[i, n] = (vectors[i, n] - alpha) / sum_n_v
    return h


def main(data):
    standardize_data = {d: [] for d in list(data.keys())}
    for d, bary in data.items():
        standardize_bary = standardize(bary)
        standardize_data[d] = standardize_bary
    return standardize_data


if __name__ == '__main__':
    tree = get_files(path)
    data = {d: np.array([np.load(os.path.join(path, d, file)) for file in tree[d]]) for d in tree.keys()}
    path = "/home/lmint/Documents/programmation/python/Optimal_Transport/deliverable_clean/test_algos/results"
    main(data)
