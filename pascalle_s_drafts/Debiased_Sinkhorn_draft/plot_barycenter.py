#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

import matplotlib.pyplot as plt

from time import time
from constants import grouping


"""
This code plots the barycenters of the noisy and non-noisy datasets
in order to compare the effectiveness of the algorithm and its parameters
to be robust to noise
"""


def plot_barycenter(qnonoise_barycenter, qnoise_barycenter, parameters):
    t = time()
    print("Plotting data...")
    if grouping == "many":
        plt.figure(1, figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(qnonoise_barycenter)
        plt.title(f"No Noise with {parameters[1]} images and epsilon = {parameters[3]}")

        plt.subplot(1,2,2)
        plt.imshow(qnoise_barycenter)
        plt.title(f"Noisy Data with {parameters[1]} images and epsilon = {parameters[3]}")
        plt.axis("off")

    
    #elif grouping == "pairs":

            
    else:
        print("Error: Please specify proper grouping in constants.py")
        sys.exit()
        
    print("DONE. t =",round(time()-t,2),"s.")    


