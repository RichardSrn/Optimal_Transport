# -*- coding: utf-8 -*-

# Author: Qi Wang, Ievgen Redko, Sylvain Takerkart


import numpy as np

import barycenter_model as model


# running with and without noise
def main():
    reg = 0.001
    nb_samples = 2
    eta = 0.1
    c = -0.5

    patterns = np.load('./artificial_data.npy')
    # patterns = np.load('./artificial_data_no_noise.npy')
    patts = patterns[:nb_samples]
    nb_samples, x_size, y_size = patts.shape
    data = patts
    data = data.reshape((-1, x_size * y_size))
    data = data.T
    data_pos = data - np.min(data)
    mass = np.sum(data_pos, axis=0).max()
    # unbalanced data
    hs = data_pos / mass
    # normalized data
    mass_hs = np.sum(hs, axis=0)
    hs_hat = hs / mass_hs

    # barycenter of TLp-BI
    bary_TLp, barys_TLp = model.tlp_bi(hs, hs_hat, x_size, y_size, reg, eta,
                                       outItermax=10, outstopThr=1e-8,
                                       inItermax=100, instopThr=1e-8,
                                       log=False)
    print('barycenter of TLp-BI finished')

    np.save("./bary_TLp.npy", bary_TLp)
    np.save("./barys_TLp.npy", barys_TLp)

    # np.save("./bary_TLp_no_noise.npy", bary_TLp)
    # np.save("./barys_TLp_no_noise.npy", barys_TLp)

    # barycenter of KBCM
    bary_KBCM = model.kbcm(hs, x_size, y_size, reg, c, q=95, numItermax=10,
                           stopThr=1e-8, log=False)
    print('barycenter of KBCM finished')

    np.save("./bary_KBCM.npy", bary_KBCM)
    # np.save("./bary_KBCM_no_noise.npy", bary_KBCM)


if __name__ == '__main__':
    main()
