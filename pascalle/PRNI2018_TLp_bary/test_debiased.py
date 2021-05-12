#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:43:16 2021

@author: bananasacks
"""


import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import ot
import warnings
import sys
import os

#CURR_DIR = os.path.dirname(os.path.realpath(__file__))

import sinkhorn_barycenters as sink


#rng = np.random.RandomState(42)
#n = 2
#index = rng.choice(np.arange(200), size=(n), replace=False)
#index.sort()
#print(index)


#can add index back here
noise15 = np.load("./artificial_data.npy")[15, :, :]
no_noise15 = np.load("./artificial_data_no_noise.npy")[15, :, :]
noise95 = np.load("./artificial_data.npy")[95, :, :]
no_noise95 = np.load("./artificial_data_no_noise.npy")[95, :, :]


#making dataset smaller
noise15 = noise15[23:46,23:46]
no_noise15 = no_noise15[23:46,23:46]
no_noise95 = no_noise95[16:39,27:50]
noise95 = noise95[16:39,27:50]


P_noise = np.array([noise15,noise95])
P_nonoise = np.array([no_noise15, no_noise95])

epsilon = 1e-1
C = ot.dist(noise15, noise95)
print(C)
#normalize
#C /= C.max()
expC = lambda i: math.exp(-i/epsilon)
expC = np.vectorize(expC)

K = expC(C)
print(K.shape)

#noiseP =  torch.Tensor(noise)
P =  torch.from_numpy(P_nonoise)
K = torch.from_numpy(K)

sink.barycenter_debiased_2d(P, K)
