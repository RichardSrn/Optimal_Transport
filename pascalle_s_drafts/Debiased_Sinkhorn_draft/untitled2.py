#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:50:25 2021

@author: bananasacks
"""

import numpy as np
import os
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")

#from load_data import load_data


#seed = 42
#parameters = np.load(" ./Data/parameters_barycenters_pairs3_0.1.npy")


#print(nonoise[1])

#CURR_DIR = os.path.dirname(os.path.realpath(__file__))

#parameters = np.load(CURR_DIR + "/Data/parameters_barycenters_pairs3_0.1.npy", allow_pickle=True)

#print(parameters)


#files = np.load(../CURR_DIR+ "/test_algos/data/artificial_data_lvl_0.000_mean_0.000.npy")

#print(files.shape )

import matplotlib.pyplot as plt
import numpy as np
import ot
from os import listdir
from os.path import isfile, join





#data = np.load("results/tlp_bary/bary_lvl_0.050_mean_0.000_reg_0.1_eta_0.1_outer-inner_10-100.npy", allow_pickle=True)
#data = data[:3] #to truncate the dataset for testing
#print(data.shape)
#print(data[0].shape)
#data = np.reshape(data, (len(data), 2500))
#print(data.shape)
#print(data[0].shape)
#print(data[0])



def get_files(noise_lvl=6):
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    if noise_lvl == 4:
        onlyfiles = [file for file in onlyfiles if file[-9:] in ["0.000.npy", "0.100.npy", "0.500.npy", "1.000.npy"]]
    elif noise_lvl == 6:
        onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]
       
    onlyfiles.sort()
    
    for file in onlyfiles:
        yield file
        
        
files = get_files()
        
        
#data = data.reshape((-1, x_size * y_size))        
        
        
        
        
        
        
    #print(data[i].ravel())
    #print(new_data.shape) 
    #print(new_data[0].shape) 
    #print(new_data[0])
    
    ##flatten the data so it is a [2500, N] array
    #print(data[0].shape)
    #print(np.prod(data[0].shape))
    #print(data[0].reshape(np.prod(data[0].shape)))
        
