#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

import sys

import numpy as np

from constants import DATAPATH_NOISE, DATAPATH_NONOISE, N, grouping
from time import time 


"""
This function takes creates an array of noisy and non-noisy randomly chosen image sets 
with the same indices between the files for easy barycenter comparison 
"""

def load_data(seed):
    t = time()
    print("loading data...")
    rng = np.random.RandomState(seed)
    if grouping == "many":
        img_index = np.array(rng.choice(np.arange(199), size=(N), replace=False), int)
        img_index.sort()
        nonoise = abs(np.load(DATAPATH_NONOISE)[img_index])
        noise = abs(np.load(DATAPATH_NOISE)[img_index])

    
    elif grouping == "pairs":
        img_index = np.empty((0,2), int)
        for i in range (N):
            index = np.array(rng.choice(np.arange(199), size=(2), replace=False), int)
            #can add code here later in case there's a repetition of the same indices (although highly unlikely)
            img_index = np.append(img_index, np.array([index]), axis=0)
            nonoise = abs(np.load(DATAPATH_NONOISE)[img_index])
            noise = abs(np.load(DATAPATH_NOISE)[img_index])
            ##This gives me an array of images, where noise[0] contains the first pair noise[0][0] and noise[0][1]
            ##which matches the same indices of pairs in the no_noise group. 
            ##The second pair being noise[1][0] and noise[1][1] and so on
            
    else:
        print("Error: Please specify proper grouping in constants.py")
        sys.exit()
        
    print("DONE. t =",round(time()-t,2),"s.")    
    return img_index, nonoise, noise

#returns the index of the images within the original datasets, 
#along with the values in noise and no noise at those indices
    

"""
#Test Function
if __name__ == "__main__":
  img_index, nonoise, noise = load_data(42)  
  
print(img_index) 
print(noise.shape)
print(noise[0][0])
"""


  

    
"""   
IGNORE    
    
 #img1, img2 = np.load(DATA_PATH)[index]

    
#load datasets, this function takes the random indexes and loads n pairs of artificial data with and without noise
def load_data(num_pairs, index):

        

#returning variables
noise, nonoise = load_data(n,index)   
    
#function to choose n random artificial datasets
def find_index(n):
    index = rng.choice(np.arange(200), size=(n), replace=False)
    index.sort()

    

#returing variables
    index = find_index(n)
    
 return print("it ran")


#here it only runs the one function within the file, to test with different outputs
if __name__ == "__main__":
    load_data(42)
"""