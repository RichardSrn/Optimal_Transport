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
import os
import warnings
import sys
import sinkhorn_barycenters as sink


#rng = np.random.RandomState(42)
#n = 2
#index = rng.choice(np.arange(200), size=(n), replace=False)
#index.sort()
#print(index)


#can add index back here
#working with 2 images noise vs no noise
noise15 = np.load("./artificial_data.npy")[15, :, :]
no_noise15 = np.load("./artificial_data_no_noise.npy")[15, :, :]
noise95 = np.load("./artificial_data.npy")[95, :, :]
no_noise95 = np.load("./artificial_data_no_noise.npy")[95, :, :]
#making dataset smaller
noise15 = noise15[20:43,20:43]
no_noise15 = no_noise15[20:43,20:43]
no_noise95 = no_noise95[19:42,27:50]
noise95 = noise95[19:42,27:50]

epsilon = .3 #1e-1
noise = np.array([noise15,noise95])
nonoise = np.array([no_noise15, no_noise95])


def debiased_format(imgs, epsilon):
  C = ot.dist(imgs[0], imgs[1])
  #normalize
  C /= C.max()
  expC = lambda i: math.exp(-i/epsilon)
  expC = np.vectorize(expC)
  K = expC(C)
  P = torch.from_numpy(imgs)
  K = torch.from_numpy(K)
  return P, K

print(noise15)