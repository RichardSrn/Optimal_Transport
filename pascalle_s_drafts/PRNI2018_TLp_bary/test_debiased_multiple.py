#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:23:30 2021

@author: bananasacks
"""


#Uses debiased sinkhorn method to calculate the barycenters for groups random of images
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
n = 3 #choosing how many datasets to use (noise, no noise) for one barycenter
epsilon = .2 #will optimize this later
max_iterations = int(1e6) #will optimize this later
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


##Runs sinkhorn algorithm for multiple data sets within noise and no noise respectively
def run_sinkhorn_mult(P_noise, K_noise, P_nonoise, K_nonoise, max_iterations):
    qnonoise = sink.barycenter(P_nonoise, K_nonoise, reference="debiased", maxiter = max_iterations)
    qnoise = sink.barycenter(P_noise, K_noise, reference="debiased", maxiter = max_iterations)
    return qnonoise, qnoise


q_nonoise_mult, q_noise_mult = run_sinkhorn_mult(P_noise, K_noise, P_nonoise, K_nonoise, max_iterations)


print(q_noise_mult)
print(index)


np.save("./index_mult3.npy", index)
np.save("./q_nonoise_mult3.npy", q_nonoise_mult)           
np.save("./q_noise_mult3.npy", q_noise_mult)

#plotting results for barycenters with multiple datasets per barycenter
q_nonoise_mult = np.load('./q_nonoise_mult3.npy')
q_noise_mult = np.load('./q_noise_mult3.npy')
index_mult = np.load('./index_mult3.npy')

#function to plot just the barycenter for multiple datasets at once
def plot_compare_mult(q_noise, q_nonoise):
    m=1
    for i in range(0,len(q_nonoise)): #change back to len(q_noise) for the def
        plt.figure(1, figsize=(10,100))
        plt.subplot(len(q_nonoise),2,i+m)
        plt.imshow(q_nonoise[i][0]) #change back to q_nonoise for the def
        plt.title(f"No Noise \n max iterations: {int(parameters[i][0])} and epsilon: {parameters[i][1]}")
        plt.axis("off")

        plt.subplot(len(q_noise),2,i+m+1)
        plt.imshow(q_noise[i][0]) #change back to q_noise for the def
        plt.title(f"Noise \n max iterations: {int(parameters[i][0])} and epsilon: {parameters[i][1]}")
        plt.axis("off")
        m+=1
        
plot_compare_mult(q_noise_mult, q_nonoise_mult) 