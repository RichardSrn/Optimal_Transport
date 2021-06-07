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
import ot
import torch

from computeK import computeK
import sinkhorn_barycenters as sink



def get_files():
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    onlyfiles = [file for file in onlyfiles if file[-5:] == "0.npy"]
    onlyfiles.sort()

    for file in onlyfiles:
        yield file
        
files = get_files()   



def debiased_sink_bary(epsilon = .1, max_iter = int(1000), plot=True, save=True):
    if plot:
        plt.figure(1, figsize=(15, 10))
        k = 1
    vmin = []
    vmax = []
    
    for file in files:
        #print(file)
        title = "bary" + file[15:-4] + "_eps_" + str(epsilon) + "_iter_" + str(max_iter)
        data = np.load("./data/" + file)
        data = data[:1] #to truncate the dataset for testing
        imgs = len(data)
        #print(len(data))
        print("./data/" + file)

        #compute P and K using computeK()
        P, K = computeK(data, epsilon)     
        #run sinkhorn algo using debiased_sinkhorn()
        bary = sink.barycenter(P, K, reference="debiased", maxiter = max_iter)  
        print(torch.min(bary))
        #if torch.min(bary) < vmin:
        #    vmin = torch.min(bary)

        print(type(bary))
        
        
        np.save("./results/debiased_sink_bary/" + title + "_imgs_" + str(imgs) + ".npy", bary)
#go through every file with a different noise level, run the algo on it, save the results
#get the vmax and vmin from the results and plot
#should I have plot as a separate function?
"""   
    
    if plot:
        plt.subplot(2, 3, k)
        plt.title(title[:len(title)//2]+"\n"+title[len(title)//2:])
        ##add vmin and vmax so all plots have same itensity scale
        plt.imshow(bary, vmin=0, vmax=1)
        k += 1
    if save:
        plt.savefig("./results/debiased_sink_bary/plots_debiased_sink_bary/" + title + ".png")
        
    if plot:
        plt.show()

"""

if __name__ == "__main__":
    debiased_sink_bary(.5, 50, save = False) #[.001, .025, .05, .075, .1, .15, .2, .3, .5, .7, .9]

#automate this laterrr
#max_iter = [100, 500, 750, 1000, 2500]
  