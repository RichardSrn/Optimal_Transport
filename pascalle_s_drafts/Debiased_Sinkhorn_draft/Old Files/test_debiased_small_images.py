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

epsilon = 1e-1
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

P_nonoise, K_nonoise = debiased_format(nonoise, epsilon)
q_nonoise = sink.barycenter_debiased_2d(P_nonoise, K_nonoise)
q_nonoise = np.array(q_nonoise)

P_noise, K_noise = debiased_format(noise, .3)
q_noise = sink.barycenter_debiased_2d(P_noise, K_noise, maxiter=1000000)
q_noise = np.array(q_noise)

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


print(P_noise[0][14])
print(q_noise[14])
print(q_nonoise[14])