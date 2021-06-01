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
    #max_iter = int(1000)
    #print(epsilon)
    if plot:
        plt.figure(1, figsize=(15, 10))
        k = 1
    
    for file in files:
        #print(file)
        title = "bary" + file[15:-4] + "_eps_" + str(epsilon)
        #print(title)
        data = np.load("./data/" + file)
        #data = data[:10] #to truncate the dataset for testing
        print("./data/" + file)

        #compute P and K using computeK()
        P, K = computeK(data, epsilon)     
        #run sinkhorn algo using debiased_sinkhorn()
        bary = sink.barycenter(P, K, reference="debiased", maxiter = max_iter)         
        np.save("./results/debiased_sink_bary/" + title + ".npy", bary)
    
        if plot:
            plt.subplot(2, 3, k)
            plt.title(title[:len(title)//2]+"\n"+title[len(title)//2:])
            plt.imshow(bary)
            k += 1
    
    if plot:
        plt.show()
    #if save:
    #    plt.savefig("./results/debiased_sink_bary/debiased_sink_"+str(epsilon)+"_eps_"+str(max_iter)+"_iter_"+str(data.shape[0])+"_samples.png")


if __name__ == "__main__":
    debiased_sink_bary(.9, 1000) #[.001, .025, .05, .075, .1, .15, .2, .3, .5, .7, .9]

#automate this laterrr
#max_iter = [100, 500, 750, 1000, 2500]
#eps = [.001, .025, .05, .075, .1, .15, .2, .3, .5, .7, .9]
    