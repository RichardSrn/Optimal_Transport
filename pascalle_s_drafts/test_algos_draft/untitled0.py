#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""


import numpy as np
import os
from os.path import isfile, join
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")
import matplotlib.pyplot as plt


def get_files():
    onlyfiles = [f for f in os.listdir("./results/kbcm_bary") if isfile(join("./results/kbcm_bary", f))]
    onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]
    #onlyfiles.sort()
    for file in onlyfiles:
        yield file
    

#add def here
        
files = get_files()       
#if plot:
plt.figure(1, figsize=(15, 10))
k = 1

for file in files:
    bary = np.load("./results/kbcm_bary/" + file, allow_pickle = True)
    title = file[:-4]
    print(title)


#need to get a unique list of parameter groupings
#then I want to plot all 6 charts (6 noise levels) in that grouping
#get a list of regs with c as second column
