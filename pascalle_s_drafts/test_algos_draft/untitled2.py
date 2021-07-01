#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""
#plot 9 rows by 5 columns
#rows are the diff params and cols are the noise levels

import os
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/deliverable_clean/test_algos")

import matplotlib.pyplot as plt
import numpy as np
import torch


def findmax(algo, reg, second_param):
    vmax=[]
    vmin=[]
    for r in reg:
        for p in second_param: 
            title = "./results/debiased_sink_bary/bary_noiselvl_"
            params = "_eps_"+str(r)+"_iter_"+str(p)+"_intensity_minmax"
                
            path = title + "0.000" + params + ".npy"
            if os.path.isfile(path):                
                noise_lvls = ["0.000", "0.100", "0.200", "0.500", "1.000"]
                for n in noise_lvls: 
                    barys = np.load(title + n + params + ".npy")
                    nanmin = np.nanmin(barys)
                    nanmax = np.nanmax(barys)
            #Finding max and min intensities for consistent plotting
                    #Finding the Min intensity with NAN handler
                    #print(vmin)
                    #print(nanmin)
                    if vmin == []:
                        if np.isnan(nanmin) == True:
                            vmin = []
                        else:
                            vmin = nanmin
                    ##If NAN, do nothing
                    ##If min(bary) > vmin, do nothing
                        #print(vmin)
                    else:
                        if np.isnan(nanmin) == False:
                            barymin = nanmin
                            if  barymin < vmin:
                                vmin = barymin 
                        #print(vmin)
                    if np.isnan(vmin):
                        vmin = 0
                    #print(vmin)
                    #Finding the Max intensity with NAN handler
                    if vmax == []:
                        if np.isnan(nanmax) == True:
                            vmax = []
                        else:
                            vmax = nanmax
                    ##If NAN, do nothing
                    ##If max(bary) < vmax, do nothing
                    else:
                        if np.isnan(nanmax) == False:
                            barymax = nanmax
                            if  barymax > vmax:
                                vmax = barymax     
                    if np.isnan(vmax):
                        vmax = 1
    return vmax
                

def bigplotkt(algo, reg, second_param, vmax):  
    plt.figure(1, figsize=(15, 10)) 
    #fig, ax = plt.subplots(10,5)
    k = 1 
    
    for r in reg:
        for p in second_param: 
            if algo == "kbcm":
                title = "./results/kbcm_bary/bary_noiselvl_"
                params = "_reg_"+str(r)+"_c_"+str(p)+"_iters_100_intensity_maxmin"
            elif algo == "tlp":
                title = "./results/tlp_bary/bary_noiselvl_"
                params = "_reg_"+str(r)+"_eta_"+str(p)+"_outer-inner_10-100_intensity_minmax"
            else:
                print("something went wrong")
            
                          
            path = title + "0.000" + params + ".npy"
            if os.path.isfile(path):
                print("_reg_"+str(r)+"_c_"+str(p))  
                
                noise_lvls = ["0.000", "0.100", "0.200", "0.500", "1.000"]
                
                #fig.suptitle(title[10:-15] + params)
                for n in noise_lvls: 
                    barys = np.load(title + n + params + ".npy")
                    #print(n)
                    plt.subplot(11, 5, k)
                    k += 1
                    plt.axis("off")
                    plt.imshow(barys, vmin=0, vmax=vmax)
                    #plt.ylabel("reg_"+str(r)+"_c_"+str(p))
                    #plt.xlabel()
                
            plt.suptitle(title[10:-15] + "     max_amplitude: "+ str(round(vmax,5)))
    
                #plt.savefig("results/report_plots/"+title[10:-15] + params + ".png")
    
    plt.show()
                
                
                
                

algo = "kbcm"
reg = [0.1, 0.01, 0.001, 0.4, 0.5, 0.05, 0.6, 0.9, 0.25, 0.75, 1, 4]
c = [-0.5, -0.7]
vmax = findmax(algo, reg, c)
print(vmax)
bigplotkt(algo, reg, c, vmax)
    
"""
algo = "tlp"
reg = [0.4, 0.05, 0.75, 1, 4]
eta = [0.1, 0.001, 0.05, 0.7]
vmax = findmax(algo, reg, eta)
print(vmax)
bigplotkt(algo, reg, eta, vmax)  
"""    
"""
            elif algo == "sink":
                title = "./results/debiased_sink_bary/bary_noiselvl_"
                params = "_eps_"+str(r)+"_iter_"+str(p)+"_intensity_minmax"
"""                
    
    