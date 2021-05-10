#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:48:50 2021

@author: bananasacks
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import ot
import warnings
import sys
sys.path.insert(1, '/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle')
from debiased_ot_barycenters import sinkhorn_barycenters as sink


rng = np.random.RandomState(42)
n = 2
    
index = rng.choice(np.arange(200), size=(n), replace=False)
index.sort()

noise = np.load("./artificial_data.npy")[index, :, :]
no_noise = np.load("./artificial_data_no_noise.npy")[index, :, :]


epsilon = 1e-1
C = ot.dist(noise[0], noise[1])
print(C)
#normalize
C /= C.max()
expC = lambda i: math.exp(-i/epsilon)
expC = np.vectorize(expC)

K = expC(C)
print(K.shape)

#noiseP =  torch.Tensor(noise)
P =  torch.from_numpy(noise)
K = torch.from_numpy(K)


#q = sink.barycenter_debiased_2d(P, K)
