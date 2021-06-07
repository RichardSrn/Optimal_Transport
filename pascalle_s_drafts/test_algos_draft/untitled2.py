#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:50:25 2021

@author: bananasacks
"""

import numpy as np
import os
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")
import matplotlib.pyplot as plt
#from load_data import load_data




#bary_lvl_0.000_mean_0.000_eps_0.7.npy
#bary_lvl_0.200_mean_0.000_eps_0.7.npy
#bary_lvl_0.050_mean_0.000_eps_0.7.npy


lvl = ["0.000", "0.050", "0.100", "0.200", "0.500", "0.100"]
eps = ["0.5"]
    

bary0 = np.load("results/debiased_sink_bary/bary_lvl_0.000_mean_0.000_eps_"+eps[0]+".npy")#, allow_pickle=True)
    
print(bary0.max())
print(bary0.min())

bary2 = np.load("results/debiased_sink_bary/bary_lvl_0.500_mean_0.000_eps_"+eps[0]+".npy")#, allow_pickle=True)
    
print(bary2.max())
print(bary2.min())
    
    #plt.imshow(bary, vmin=0, vmax=1)

#k=1
#plt.subplot(2, 3, k)
#plt.title(title[:len(title)//2]+"\n"+title[len(title)//2:])
#plt.imshow(bary, vmin=0, vmax=1)
#k += 1

        

        
