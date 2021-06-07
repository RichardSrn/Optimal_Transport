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

from barycenter_model import tlp_bi



def get_files():
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    onlyfiles = [file for file in onlyfiles if file[-5:] == "0.npy"]
    onlyfiles.sort()

    for file in onlyfiles:
        yield file
  
#parameters
#    reg : float
#        Entropic regularization term
#    eta : float
#        The parameter for cost matrix
#    outItermax: int
#        Max number of iterations for outer loop
#    inItermax: int
#        Max number of iterations for inner loop
#    outstopThr : float
#        Stop threshold for outer loop
#    instopThr : float
#        Stop threshold for inner loop 
###
        
def tlp_bary(reg = 0.1, eta = 0.1, x_size = 50, y_size = 50, outItermax=10, weights=None, inItermax=100, outstopThr=1e-8, instopThr=1e-8, log=False, plot=True, save=True):      

    files = get_files()       
    if plot:
        plt.figure(1, figsize=(15, 10))
        k = 1
    
    for file in files:
        title = "bary" + file[15:-4] + "_reg_" + str(reg) + "_eta_" + str(eta)+ "_outer-inner_" + str(outItermax)+"-"+str(inItermax)
        data = np.load("./data/" + file)
        data = data[:5] #to truncate the dataset for testing
        data = np.reshape(data, (len(data), 2500))   
        #data = data.reshape((-1, x_size * y_size))
        print(data[0])
        #data = data.reshape((-1, x_size * y_size))
        data = data.T
        data_pos = data - np.min(data)
        mass = np.sum(data_pos, axis=0).max()
        # unbalanced data
        hs = data_pos / mass
        # normalized data
        mass_hs = np.sum(hs, axis=0)
        hs_hat = hs / mass_hs

        # barycenter of tlp_bi
        bary, barys = tlp_bi(hs, hs_hat, x_size, y_size, reg, eta, weights, outItermax, inItermax, outstopThr, instopThr, log=log)
        print(bary[0])
        bary = np.reshape(bary, (50,50)) 
        #print(bary[0])


        np.save("./results/tlp_bary/" + title + ".npy", bary)
    
        if plot:
            plt.subplot(2, 3, k)
            plt.title(title[:len(title)//2]+"\n"+title[len(title)//2:])
            plt.imshow(bary)
            k += 1
    #if save:
    #    plt.savefig("./results/tlp_bary/tlp_"+str(reg)+"_reg_"+str(eta)+"_eta_"+str(data.shape[0])+"_samples.png")   
    if plot:
        plt.show()

 

if __name__ == "__main__":
    reg = [.001, .01, .05, .1, .5, .9]
    eta = [.001, .1]#, .05, .7]
    for r in reg: 
        for e in eta:
            tlp_bary(reg = r, eta = e)#, outItermax = 20, inItermax=100, outstopThr=1e-10, instopThr=1e-10)
    
    
    
    
#    reg = [.05] #[.01, .05, .1, .5]
#    for r in reg:
#        tlp_bary(reg = r)
            
        
        
        
        