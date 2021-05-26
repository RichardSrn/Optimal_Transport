#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

import sys

import numpy as np
import torch

from constants import DATAPATH_NOISE, DATAPATH_NONOISE, N, grouping, epsilon, width
from time import time 

"""
This function creates and formats the inputs for the Debiased Sinkhorn Algorithm

P is a tensor object of all the images or image sets
K is the 50x50 manipulation of the cost matrix M to use as input for the sinkhorn algo
"""


#formatting function
def computeK(imgset):
    t = time()
    print("computing K...")
    P = torch.from_numpy(imgset)    
    grid = torch.arange(width).type(torch.float64)
    grid /= width #can automate width, but for now it is in constants.py
    M = (grid[:, None] - grid[None, :]) ** 2
    K = torch.exp(- M / epsilon)
    #K = torch.from_numpy(K)
    print("DONE. t =",round(time()-t,2),"s.")   
    return P, K



"""
#Test Function
if __name__ == "__main__":
  P, K = computeK()  
  
print(P) 
print(P.shape)
print(K)
print(K.shape)
"""


"""
Note to self, ask richard if he agrees with how M is computed here and how it relates to Qi's method
"""