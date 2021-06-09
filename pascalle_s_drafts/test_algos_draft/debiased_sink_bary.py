#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

#! /usr/bin/env python3
import os
from os import listdir
from os.path import isfile, join
#os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")

import matplotlib.pyplot as plt
import numpy as np
import torch

from computeK import computeK
import sinkhorn_barycenters as sink



"""
intensity = "minmax" or "zeroone"
    Will set the plots based off of the minimum and maximum intensity values across all noise levels
    Or sets the plot's intensity to be between 0 and 1
"""

def get_files():
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    onlyfiles = [file for file in onlyfiles if file[-5:] == "0.npy"]
    onlyfiles.sort()

    for file in onlyfiles:
        yield file
        
files = get_files()   



def debiased_sink_bary(epsilon = .1, max_iter = int(1000), intensity = "zeroone", noise_lvl=6, imgs = 5, plot=True, save=True):
    if plot:
        plt.figure(1, figsize=(15, 10))
    vmin = []
    vmax = []
    
    for file in files:
        #print(file)
        data = np.load("./data/" + file)
        data = data[:imgs] #to truncate the dataset for testing
        #print("./data/" + file)

        #compute P and K using computeK()
        P, K = computeK(data, epsilon)     
        #run sinkhorn algo using debiased_sinkhorn()
        bary = sink.barycenter(P, K, reference="debiased", maxiter = max_iter)  
        if vmin == []:
            vmin = torch.min(bary)
        elif vmin < torch.min(bary):
            vmin = torch.min(bary)
        if vmax == []:
            vmax = torch.max(bary)
        elif vmax > torch.max(bary):
            vmax = torch.max(bary)
            
        
        title = "bary" + file[15:-4] 
        params = "_eps_" + str(epsilon) + "_iter_" + str(max_iter) + "_imgs_" + str(imgs) + "_intensity_" + str(intensity)
        np.save("./results/debiased_sink_bary/" + title + params + ".npy", bary)
 #need to put in a handler for NAN  
    print(vmin)
    print(vmax)     
    if np.isnan(vmin) == 0:
        vmin = vmin.numpy()
    if np.isnan(vmax) == 0:
        vmax = vmax.numpy()

        
#go through every file with a different noise level, run the algo on it, save the results
#get the vmax and vmin from the results and plot
#should I have plot as a separate function?
  
  
    if plot:
        k = 1
        params = "_eps_" + str(epsilon) + "_iter_" + str(max_iter) + "_imgs_" + str(imgs) + "_intensity_" + str(intensity)
        if noise_lvl == 6:
            noise_lvls = ["0.000", "0.050", "0.100", "0.200", "0.500", "1.000"]
            m = 3
            #artificial_data_lvl_0.200_mean_0.000.npy
        elif noise_lvl == 4:
            noise_lvls = ["0.000", "0.100", "0.500", "1.000"]
            m = 2
            #artificial_data_noiselvl_0.500.npy
        for n in noise_lvls: 
            #bary_lvl_0.500_mean_0.000_eps_0.5_iter_50_imgs_5_intensity_zeroone.npy
            barys = np.load("./results/debiased_sink_bary/bary_lvl_"+n+"_mean_0.000"+params+".npy")
            plt.subplot(2, m, k)
            plt.title("bary_lvl_"+n+"_mean_0.000")
            ##added vmin and vmax so all plots have same itensity scale
            k += 1
            if intensity == "zeroone":
                plt.imshow(barys, vmin=0, vmax=1)
            elif intensity == "minmax":
                plt.imshow(barys, vmin=vmin, vmax=vmax)
            else:
                plt.imshow(barys)
        

    if save:
        plt.savefig("./results/debiased_sink_bary/plots_debiased_sink_bary/" + title + params + ".png")
        
    if plot:
        plt.show()



if __name__ == "__main__":
    #iters = [100, 750, 2000, 10000, 1e8]
    #for i in iters:
    debiased_sink_bary(epsilon = .5, max_iter = 100, intensity = "minmax", noise_lvl=4, save=False) 



#takes a long time to run
#epsilon = .0001, max_iter = int(1e6)
#epsilon = .05, max_iter = int(1e7)

#eps = [.0001, .005, .01, .05, .1, .2, .5, .7]    
#iters = [100, 750, 2000, 1e8]
    
#[.001, .025, .05, .075, .1, .15, .2, .3, .5, .7, .9]


#ALREADY RUN
#eps = .05 and iters = [ ] (try with high iterations like 1e6,8,10  [[100, 500, 1000, 2000, 3000, 5000, 1e5] don't work
#eps = .1 and iters = [100, 10000, ] Need more than 10000 for .1
#eps = .2 and iters = [2500, 5000] lower than 2500 is no good, 5000 is no good
#eps = .5 and iters = [500, 750, 1000]
#eps = .7 and iters = [100, 750, 1000]
#automate this laterrr
#max_iter = [100, 500, 750, 1000, 2500]
  