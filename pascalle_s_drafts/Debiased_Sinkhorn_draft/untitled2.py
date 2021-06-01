#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:50:25 2021

@author: bananasacks
"""

import numpy as np
import os
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")
from debiased_sink_bary import debiased_sink_bary
#from load_data import load_data


#seed = 42
#parameters = np.load(" ./Data/parameters_barycenters_pairs3_0.1.npy")


#print(nonoise[1])

#CURR_DIR = os.path.dirname(os.path.realpath(__file__))

#parameters = np.load(CURR_DIR + "/Data/parameters_barycenters_pairs3_0.1.npy", allow_pickle=True)

#print(parameters)


#files = np.load(../CURR_DIR+ "/test_algos/data/artificial_data_lvl_0.000_mean_0.000.npy")

#print(files.shape )

#eps = [.001, .025, .05, .075, .1, .15, .2, .3, .5, .7 ]
 
#for e in eps:
#    debiased_sink_bary(e) 

max_iter = [100, 500, 750, 1000, 2500, 3000, 5000]
eps = [.001, .025, .05, .075, .1, .15, .2, .3, .5, .7 ]
 
for m in max_iter:
    print(m)
    for e in eps:
        print(e)
        debiased_sink_bary(e, m) 