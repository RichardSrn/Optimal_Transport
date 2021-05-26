#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:44:07 2021

@author: bananasacks
"""


#Uses debiased sinkhorn method to compare random PAIRS of images
#to visualize and determine robustness to noise

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import ot
import os
import warnings
import sys
import sinkhorn_barycenters as sink
from itertools import combinations

#set variables
n = 3 #choosing n pairs (noise, no noise) of random datasets
epsilon = 1e-1 #will optimize this later
max_iterations = 30000 #will optimize this later
width = 50
rng = np.random.RandomState(42)#42


#function to choose n random artificial datasets
def find_index(n):
    index = rng.choice(np.arange(200), size=(n), replace=False)
    index.sort()
    return index

#returing variables
index = find_index(n)


#load datasets, this function takes the random indexes and loads n pairs of artificial data with and without noise
def load_data(num_pairs, index):
    noise = []
    nonoise = []
    for i in range(len(index)):
        noise.append(abs(np.load("./artificial_data.npy")[index[i], :, :]))
        nonoise.append(np.load("./artificial_data_no_noise.npy")[index[i], :, :])
    return np.asarray(noise), np.asarray(nonoise)
        

#returning variables
noise, nonoise = load_data(n,index) 



#formatting function
def debiased_format(dataset, epsilon):
    P = torch.from_numpy(dataset)    
    grid = torch.arange(width).type(torch.float64)
    grid /= width
    M = (grid[:, None] - grid[None, :]) ** 2
    K = torch.exp(- M / epsilon)
    #K = torch.from_numpy(K)
    return P, K

#creating variables for debiased algo
P_nonoise, K_nonoise = debiased_format(nonoise, epsilon)
P_noise, K_noise = debiased_format(noise, epsilon)

print(P_nonoise.shape)
print(K_nonoise.shape)


##Runs every combination of pairs for data sets within noise and no noise respectively

def run_sinkhorn(n, index, max_iterations, P_noise, K_noise, P_nonoise, K_nonoise):
    #row_num = int((n*(n-1))/2)
    q_nonoise = []
    q_noise = []
    index_list = []
    for i in range(len(index)):
        index_list.append(i)
    #list of all unique combinations of pairs of indices of datasets    
    combo_list = list(combinations(index_list,2)) 

    for i in range(len(combo_list)):
        first_set = combo_list[i][0]
        second_set = combo_list[i][1]
        #number for the image corresponding to the dataset
        image_nums = [index[first_set],index[second_set]] 
        #getting indices for all possible combinations of pairs
        ind = torch.tensor([first_set, second_set])
        #all possible combinations of pairs of images with no noise
        Pnonoise = torch.index_select(P_nonoise, 0, ind)
        qnonoise = sink.barycenter(Pnonoise, K_nonoise, reference="debiased", maxiter = max_iterations)
        q_nonoise.append(qnonoise)

        #all possible combinations of pairs of images with noise
        Pnoise = torch.index_select(P_noise, 0, ind)
        qnoise = sink.barycenter(Pnoise, K_noise, reference="debiased", maxiter = max_iterations)
        q_noise.append(qnoise)
        
    return combo_list, image_nums, q_nonoise, q_noise

#combo_list, image_nums, q_nonoise, q_noise = run_sinkhorn(n, index, max_iterations, P_noise, K_noise, P_nonoise, K_nonoise)        



#plotting just the barycenters
##NEED to add a way to the plots to see which images aren't working (or get nans)
#also should see if I can optimize the regularization

def plot_compare(q_noise, q_nonoise):
    for i in range(0,len(q_noise)):
        print(i)
        plt.figure(i+1, figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(q_nonoise[i])
        plt.title("No Noise")

        plt.subplot(1,2,2)
        plt.imshow(q_noise[i])
        plt.title("Noise")
        plt.axis("off")

#plot_compare(q_noise, q_nonoise)   


           
max_iterations = [1e4] #, 1e10, 1e11] #[1e4, 1e2] 
epsilon = [5e-3, 1e-1, .2, .5, .7] #[1e-1, .3] 
parameters = []    
qno = []
qyes = []
           
for m in max_iterations:
    for e in epsilon:
        m = int(m)
        e = float(e)
        print("another one")
        parameters.append([m,e])
        P_nonoise, K_nonoise = debiased_format(nonoise, e)
        P_noise, K_noise = debiased_format(noise, e)
        combo_list, image_nums, q_nonoise, q_noise = run_sinkhorn(n, index, m, P_noise, K_noise, P_nonoise, K_nonoise)        
        qno.append(q_nonoise)
        qyes.append(q_noise)
        
        
np.save("./qnonoise_barycenters4.npy", qno)
np.save("./qwithnoise_barycenters4.npy", qyes)           
np.save("./parameters_barycenters4.npy", parameters)         


