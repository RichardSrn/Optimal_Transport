#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""
import os

from os import listdir
from os.path import isfile, join

#os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")

import matplotlib.pyplot as plt
import numpy as np
import ot

from barycenter_model import kbcm



def get_files():
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    onlyfiles = [file for file in onlyfiles if file[-5:] == "0.npy"]
    onlyfiles.sort()

    for file in onlyfiles:
        yield file
  
#parameters
#    reg : float
#       Entropic regularization term 
#    c : float
#       Step size for gradient update    

"""
explain intensity
imgs
automate xsize ysize and take out of function
"""

def kbcm_bary(reg = 0.001, c = -0.5, x_size = 50, y_size = 50, max_iter= 500, intensity = "zeroone", noise_lvl=6, imgs = 5, plot=True, save=True):      

    files = get_files()       
    if plot:
        plt.figure(1, figsize=(15, 10))
        k = 1
    vmin = []
    vmax = []
    
    for file in files:
        
        data = np.load("./data/" + file)
        data = data[:imgs] #to truncate the dataset for testing
        
        data = np.reshape(data, (len(data), 2500))    
        data = data.T
        data_pos = data - np.min(data)
        mass = np.sum(data_pos, axis=0).max()
        # unbalanced data
        hs = data_pos / mass
        # barycenter of KBCM
        bary = kbcm(hs, x_size, y_size, reg, c, numItermax=max_iter)
        bary = np.reshape(bary, (50,50)) 
        
       
        if vmin == []:
            vmin = np.amin(bary)
        elif vmin > np.amin(bary):
            vmin = np.amin(bary)
        if vmax == []:
            vmax = np.amax(bary)
        elif vmax < np.amax(bary):
            vmax = np.amax(bary)
        #print(vmin, "min")
        #print(vmax, "max")
                  
        title = "bary" + file[15:-4] 
        params = "_reg_" + str(reg) + "_c_" + str(c) + "_iters_" + str(max_iter) + "_imgs_" + str(imgs) + "_intensity_" + str(intensity) + "_noise_lvls_" + str(noise_lvl)
        if save:
            np.save("./results/kbcm_bary/" + title + params + ".npy", bary)
    
    
    #print(vmin)
    #print(vmax) 
    #if np.isnan(vmin) == 0:
    #    vmin = vmin.numpy()
    #if np.isnan(vmax) == 0:
    #    vmax = vmax.numpy()
        
        
        
        
        
        
        
        
  #      if plot:
  #          plt.subplot(2, 3, k)
  #          plt.title(title[:len(title)//2]+"\n"+title[len(title)//2:])
  #          plt.imshow(bary)
  #          k += 1
            
 

    #if save:
    #    plt.savefig("./results/kbcm_bary/kbcm_"+str(reg)+"_reg_"+str(c)+"_c_"+str(data.shape[0])+"_samples.png")    
    #if plot:
    #    plt.show()

 
"""
kbcm_0.4_reg_-0.5_c_300iters_5_samples.png
     ^^^change reg parameter
"""
if __name__ == "__main__":
    reg = [.2]
    for r in reg:
        kbcm_bary(reg = r, max_iter = 10, save=False)
 
 #   [.025, .05, .1]
    
    
    
    
    
    
    
#run overnight       
#reg = .001 is taking a very long time to run
#reg = .01 is taking along time to run        

#already finished
#reg = .1 with max_iter=100     c =-.5
#reg = [.05,.5,.9] with max_iter=100  c =-.5    
#reg = [.25,.4,.6] with max_iter=100  c =-.5    
 

      
"""       
if __name__ == "__main__":
    reg = [.001, .05, .1, .5, .9]
    max_iter = [100, 500, 750, 1000, 5000]
    cs = [.001, .01, .1, .5, .7, .9]
    for m in max_iter:
        for r in reg:
            for c in cs:
                kbcm_bary(reg = r, c=-c, max_iter = m)

#reg = [.001, .025, .05, .075, .1, .15, .2, .3, .5, .7, .9]
#max_iter = [100, 500, 750, 1000, 2500, 5000]
#cs = [.001, .01, .1, .5, .7, .9]

"""
    
