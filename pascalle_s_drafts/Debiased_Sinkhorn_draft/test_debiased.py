#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:00:15 2021

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

#Loading 2 sample images
noise15 = abs(np.load("./artificial_data.npy")[15, :, :])
no_noise15 = np.load("./artificial_data_no_noise.npy")[15, :, :]
noise95 = abs(np.load("./artificial_data.npy")[95, :, :])
no_noise95 = np.load("./artificial_data_no_noise.npy")[95, :, :]

#constants
width = 50
epsilon = 1e-1

noise = np.array([noise15,noise95])
nonoise = np.array([no_noise15, no_noise95])

#formatting function
def debiased_format(imgs, epsilon):
    P = torch.from_numpy(imgs)
    
    grid = torch.arange(width).type(torch.float64)
    grid /= width
    M = (grid[:, None] - grid[None, :]) ** 2
    K = torch.exp(- M / epsilon)
    #K = torch.from_numpy(K)
    return P, K

#creating variables for debiased algo
P_nonoise, K_nonoise = debiased_format(nonoise, epsilon)
P_noise, K_noise = debiased_format(noise, epsilon)

#running debiased algo
q_noise = sink.barycenter(P_noise, K_noise, reference="debiased", maxiter = 15000)
q_nonoise = sink.barycenter(P_nonoise, K_nonoise, reference="debiased", maxiter = 15000)

#plotting results
plt.figure(1, figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(noise15)
plt.title("Noisy Image 15")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(no_noise15)
plt.title("Non-Noisy Image 15")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(noise95)
plt.title("Noisy Image 95")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(no_noise95)
plt.title("Non-Noisy Image 95")
plt.axis('off')



plt.figure(2, figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(q_noise)
plt.title("Debiased Barycenter with Noise")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(q_nonoise)
plt.title("Debiased Barycenter with No Noise")
plt.axis('off')


#print(P_noise[0][14])
#print(q_noise[0])
#print(q_nonoise[0])
