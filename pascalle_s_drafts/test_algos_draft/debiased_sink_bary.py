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
    In case of NAN values? Set to 0 and 1
    
imgs: number of images used to compute the barycenter

noise_lvl: choosing to use 4 different or 6 different noise levels for barycenters and plots
"""



def get_files(noise_lvl=6):
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    if noise_lvl == 4:
        onlyfiles = [file for file in onlyfiles if file[-9:] in ["0.000.npy", "0.100.npy", "0.500.npy", "1.000.npy"]]
    elif noise_lvl == 6:
        onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]
       
    onlyfiles.sort()
    
    for file in onlyfiles:
        yield file
        


def debiased_sink_bary(epsilon = .1, max_iter = int(1000), intensity = "zeroone", noise_lvl=6, imgs = 5, plot=True, save=True):
    
    files = get_files(noise_lvl) 
    
    if plot:
        plt.figure(1, figsize=(15, 10))
    vmin = []
    vmax = []
 
    
    for file in files:    
        data = np.load("./data/" + file)
        data = data[:imgs] ##number of images to use to compute barycenter
       

        #Computing barycenter
        P, K = computeK(data, epsilon)     
        bary = sink.barycenter(P, K, reference="debiased", maxiter = max_iter) 

        #Finding max and min intensities for consistent plotting
        #Finding the Min intensitywith NAN handler
        if vmin == []:
            if torch.isnan(torch.min(bary)) == True:
                vmin = []
            else:
                vmin = torch.min(bary)
                vmin = vmin.numpy()
        ##If NAN, do nothing
        ##If min(bary) > vmin, do nothing
        else:
            if torch.isnan(torch.min(bary)) == False:
                barymin = torch.min(bary)
                barymin = barymin.numpy()
                if  barymin < vmin:
                    vmin = barymin    
        if np.isnan(vmin):
            vmin = 0
        
        #Finding the Max intensity with NAN handler
        if vmax == []:
            if torch.isnan(torch.max(bary)) == True:
                vmax = []
            else:
                vmax = torch.max(bary)
                vmax = vmax.numpy()
        ##If NAN, do nothing
        ##If max(bary) < vmax, do nothing
        else:
            if torch.isnan(torch.max(bary)) == False:
                barymax = torch.max(bary)
                barymax = barymax.numpy()
                if  barymax > vmax:
                    vmax = barymax     
        if np.isnan(vmax):
            vmax = 1
       
        #saving the dataset
        title = "bary" + file[15:-4] 
        params = "_eps_" + str(epsilon) + "_iter_" + str(max_iter) + "_imgs_" + str(imgs) + "_intensity_" + str(intensity) + "_noise_lvls_" + str(noise_lvl)
        if save:
            np.save("./results/debiased_sink_bary/" + title + params + ".npy", bary)

        

  
##Plotting set up for either 4 or 6 groups of noise levels
#Choose the intensity scale on the plot to be eith 0/1 or min/max 
    if plot:
        k = 1
        params = "_eps_" + str(epsilon) + "_iter_" + str(max_iter) + "_imgs_" + str(imgs) + "_intensity_" + str(intensity) + "_noise_lvls_" + str(noise_lvl)
        if noise_lvl == 6:
            noise_lvls = ["0.000", "0.050", "0.100", "0.200", "0.500", "1.000"]
            m = 3
        elif noise_lvl == 4:
            noise_lvls = ["0.000", "0.100", "0.500", "1.000"]
            m = 2
        for n in noise_lvls: 
            barys = np.load("./results/debiased_sink_bary/bary_noiselvl_" + n + params + ".npy")
            plt.subplot(2, m, k)
            plt.title("bary_lvl_" + n + "_mean_0.000")
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
    debiased_sink_bary(epsilon = .6, max_iter = 100, intensity = "minmax", noise_lvl = 6) 


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


  