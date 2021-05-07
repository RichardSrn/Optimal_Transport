#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:54:35 2021

@author: bananasacks
"""


import math
import matplotlib.pyplot as plt
import numpy as np


#run create_artificial_data and create_artificial_data_no_noise using the already saved off amplitude.npy file
#run barycenter_example on noise and no_noise
#plot noise against no noise





barys_TLp = np.load("./barys_TLp.npy")
bary_KBCM = np.load("./bary_KBCM_no_noise.npy")
#bary_KBCM = np.load("./bary_KBCM.npy")
#bary_TLp = np.load("./bary_TLp.npy")





rng = np.random.RandomState(42)

n = 8

index = rng.choice(np.arange(200), size=(n), replace=False)
index.sort()

data = np.load("./artificial_data.npy")[index, :, :]
data_no_noise = np.load("./artificial_data_no_noise.npy")[index, :, :]


print("ok")
print(barys_TLp.shape)
print(bary_KBCM.shape)
#print(bary_TLp.shape)