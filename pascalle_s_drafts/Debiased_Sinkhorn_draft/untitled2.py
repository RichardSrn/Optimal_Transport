#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:50:25 2021

@author: bananasacks
"""

import numpy as np
import os
from load_data import load_data


#seed = 42
#parameters = np.load(" ./Data/parameters_barycenters_pairs3_0.1.npy")


#print(nonoise[1])

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

parameters = np.load(CURR_DIR + "/Data/parameters_barycenters_pairs3_0.1.npy", allow_pickle=True)

print(parameters)
