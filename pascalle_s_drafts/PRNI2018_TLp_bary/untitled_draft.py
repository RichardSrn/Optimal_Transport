#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:54:35 2021

@author: bananasacks
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
import sys
#sys.path.insert(1, '/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle')
#from debiased_ot_barycenters import sinkhorn_barycenters as sink


P_nonoise, K_nonoise = debiased_format(nonoise, epsilon)

q_nonoise = sink.barycenter_debiased_2d(P_nonoise, K_nonoise)

P_noise, K_noise = debiased_format(noise, epsilon)
q_noise = sink.barycenter_debiased_2d(P_noise, K_noise, maxiter=10000)

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






noise15 = np.load("./artificial_data.npy")[15, :, :]
no_noise15 = np.load("./artificial_data_no_noise.npy")[15, :, :]

noise95 = np.load("./artificial_data.npy")[95, :, :]
no_noise95 = np.load("./artificial_data_no_noise.npy")[95, :, :]

noise15 = noise15[23:46,23:46]
no_noise15 = no_noise15[23:46,23:46]
print(no_noise15.shape)

no_noise95 = no_noise95[16:39,27:50]
noise95 = noise95[16:39,27:50]


#rng = np.random.RandomState(42)
#n = 2
    
#index = rng.choice(np.arange(200), size=(n), replace=False)
#index.sort()

#noise = np.load("./artificial_data.npy")#[index, :, :]
#no_noise = np.load("./artificial_data_no_noise.npy")#[index, :, :]

#noise = noise.reshape((50,50))
#print(noise.shape)
#print(no_noise.shape)

#print(type(noise))
#noiseP =  torch.Tensor(noise)
#noiseP =  torch.from_numpy(noise)
#K = torch.from_numpy(np.array([50]))
#print(type(noiseP))

#sink.barycenter_debiased_2d(noiseP, noiseP)
#sink.barycenter_3d(noiseP, noiseP, debiased = True)


#run create_artificial_data and create_artificial_data_no_noise using the already saved off amplitude.npy file
#run barycenter_example on noise and no_noise
#plot noise against no noise





#barys_TLp = np.load("./barys_TLp.npy")
#bary_KBCM = np.load("./bary_KBCM_no_noise.npy")
#bary_KBCM = np.load("./bary_KBCM.npy")
#bary_TLp = np.load("./bary_TLp.npy")





#rng = np.random.RandomState(42)

#n = 8

#index = rng.choice(np.arange(200), size=(n), replace=False)
#index.sort()

#data = np.load("./artificial_data.npy")[index, :, :]
#data_no_noise = np.load("./artificial_data_no_noise.npy")[index, :, :]


#print("ok")
#print(barys_TLp.shape)
#print(bary_KBCM.shape)
#print(bary_TLp.shape)