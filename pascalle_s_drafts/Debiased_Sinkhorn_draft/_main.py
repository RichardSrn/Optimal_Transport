#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

import sys

import numpy as np

from load_data import load_data
from time import time 
from computeK import computeK 
from debiased_sinkhorn import debiased_sinkhorn
from constants import N, grouping, epsilon, max_iter
from plot_barycenter import plot_barycenter


#add a save = t/f
def main(seed = 42): #img1=None, img2=None, save=False, show=False, plot_title=None, absolute = True
    t = time()
    
    # get the corresponding image sets with and without noise
    img_index, nonoise, noise = load_data(seed)
    #print(img_index)
    #print(img_index[0])


    for i in range(len(epsilon)):
        print(i)
        #compute the cost matrices and K's to insert into sinkhorn algo
        Pnonoise, Knonoise = computeK(nonoise, epsilon[i])
        Pnoise, Knoise = computeK(noise, epsilon[i])  
        #run sinkhorn algorithm to calculate barycenters
        qnonoise_barycenter, qnoise_barycenter = debiased_sinkhorn(img_index, Pnonoise, Knonoise, Pnoise, Knoise)
        #save data files
        parameters = [grouping, N, img_index, epsilon[i], max_iter]
        np.save(f"./Data/qnonoise_barycenters_{grouping}{N}_{epsilon[i]}.npy", qnonoise_barycenter)
        np.save(f"./Data/qwithnoise_barycenters_{grouping}{N}_{epsilon[i]}.npy", qnoise_barycenter)           
        np.save(f"./Data/parameters_barycenters_{grouping}{N}_{epsilon[i]}.npy", parameters) 
        
        
        #plot barycenters and save
        plot_barycenter(qnonoise_barycenter, qnoise_barycenter, parameters, i)
            

    return #parameters, qnoise_barycenter, qnonoise_barycenter


main()

  
"""
ANALYZING RESULTS

print(img_index)
print(qnonoise.shape)
print(qnoise.shape)

parameters, qnoise_barycenter, qnonoise_barycenter = main()

print(parameters)
print(qnonoise_barycenter)
print(qnoise_barycenter)
#print(qnonoise.shape)
#print(qnoise.shape)
#print(qnonoise)
#print(qnoise)
"""    

#keep functions outside of other functions for reachability    
    
    
    
#Tech with Tim youtube
#can use classes to have good coding practices    

