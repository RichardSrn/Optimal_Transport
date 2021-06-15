# -*- coding: utf-8 -*-

# Author: Qi Wang, Ievgen Redko, Sylvain Takerkart

import numpy as np
from scipy.stats import multivariate_normal

def get_title(lvl, mean, sign_digits):
    if lvl == int(lvl):
        lvl = str(int(lvl))+'.'+'0'*(sign_digits-2)
    else :
        lvl = str(lvl)
        while len(lvl) < sign_digits :
            lvl = lvl+'0'
    title = './artificial_data_noiselvl_'+lvl+'.npy'
    return title

def generate_data(seed=42, nb_samples = 200,
                  x_size=50, y_size = 50,
                  noise_level = 0, noise_mean = 0):
    
    # activated zone
    #original code, but gives us new random samples each time, no way to add a seed so saving a random sample file to use 
    #for the future to keep noise and non noise consistent

    rng = np.random.RandomState(seed)

    amplitudes = rng.normal(5, 1, nb_samples)

    signalN = 11
    mean = 0
    square_sig = 0.1
    X = np.linspace(-0.5, 0.5, signalN)
    Y = np.linspace(-0.5, 0.5, signalN)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2, ))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    signal_matrix = np.empty((nb_samples, signalN, signalN))
    for i in range(nb_samples):
        mu = np.array([mean, mean])
        Sigma = np.array([[square_sig, 0], [0, square_sig]])
        F = multivariate_normal(mu, Sigma, seed=seed)
        Z = F.pdf(pos)
        Z = Z / Z.max() * amplitudes[i]
        signal_matrix[i] = Z


    # coordinates of the center of the activated zone
    coors = []
    for i in range(nb_samples//2):
        t = 2 * np.pi * rng.uniform(0, 1)
        r = np.sqrt(rng.uniform(0, 225))
        coor = [r * np.cos(t), r * np.sin(t)]
        coors.append(coor)
        coors.append([-coor[0],-coor[1]])
    coors = np.vstack(coors)
    coors += np.array([25, 25])
    coors = coors.astype(int)

    # images with signal and (maybe) noise
    gap = int(np.floor(signalN / 2))
    signals = signal_matrix
    noisefrees = np.zeros([nb_samples, x_size, y_size])
    patterns = np.zeros([nb_samples, x_size, y_size])
    for sample_id in range(nb_samples):
        loc = coors[sample_id]
        noisefrees[sample_id][loc[0] - gap:loc[0] + gap + 1, loc[1] - gap:loc[1] + gap + 1] \
            = signals[sample_id]
        patterns[sample_id] = noisefrees[sample_id] + \
                              rng.normal(noise_mean, noise_level,
                                               (x_size, y_size))

    title = get_title(noise_level, noise_mean, 5)
    np.save("./data/"+title, patterns)
    return patterns


if __name__=="__main__":
    for i in [0,0.1,0.2,0.5,1] :
        generate_data(noise_level = i)