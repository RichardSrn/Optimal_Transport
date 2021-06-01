#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

#! /usr/bin/env python3

from os import listdir
from os.path import isfile, join

os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos")

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



reg = 0.04 
epsilon = .1    
max_iter = int(1e3)

for file in files:
    #print(file)
    title = "bary" + file[15:-4] + "_reg_" + str(reg)
    #print(title)
    data = np.load("./data/" + file)
    print("./data/" + file)
    print(data.shape)
    #compute P and K using computeK()
    P, K = computeK(data, epsilon)  
    
    #run sinkhorn algo using debiased_sinkhorn()
    q = sink.barycenter(P, K, reference="debiased", maxiter = max_iter)
     

   # np.save("./results/debiased_sink_bary/" + title + ".npy", bary)

   # if plot:
   #     plt.subplot(2, 3, k)
   #     plt.title(title[:len(title)//2]+"\n"+title[len(title)//2:])
   #     plt.imshow(bary)
   #     k += 1

#if plot:
#    plt.show()
#if save:
    #fix this title later
#    plt.savefig("./results/debiased_sink_bary/entropic_"+str(reg)+"_reg_"+str(data.shape[0])+"_samples.png")


#if __name__ == "__main__":
#    entropic_reg_bary()

    