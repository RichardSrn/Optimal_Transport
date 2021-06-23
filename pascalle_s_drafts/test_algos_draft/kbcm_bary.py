#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""
import os

from os import listdir
from os.path import isfile, join
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")

import matplotlib.pyplot as plt
import numpy as np

from barycenter_model import kbcm

"""
intensity = "minmax" or "zeroone"
    Will set the plots based off of the minimum and maximum intensity values across all noise levels
    Or sets the plot's intensity to be between 0 and 1
    In case of NAN values? Set to 0 and 1
    
imgs: number of images used to compute the barycenter

noise_lvl: choosing to use 4 different or 6 different noise levels for barycenters and plots

"""

#parameters
#    reg : float
#       Entropic regularization term 
#    c : float
#       Step size for gradient update    


def get_files():
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]       
    onlyfiles.sort()
    
    for file in onlyfiles:
        yield file
  


"""automate xsize ysize and take out of function"""
def kbcm_bary(reg = 0.001, c = -0.5, x_size = 50, y_size = 50, max_iter= 500, intensity = "zeroone", plot=True, save=True):      

    files = get_files()       
    if plot:
        plt.figure(1, figsize=(15, 10))
        k = 1
    vmin = []
    vmax = []
    
    
    for file in files:
        
        data = np.load("./data/" + file)
        data = abs(data[:]) ##number of images to use to compute barycenter
        #print(data[0])
        #Computing barycenter
        data = np.reshape(data, (len(data), (x_size*y_size)))    
        data = data.T
        data_pos = data - np.min(data)
        mass = np.sum(data_pos, axis=0).max()
        # unbalanced data
        hs = data_pos / mass
        # barycenter of KBCM
        bary = kbcm(hs, x_size, y_size, reg, c, numItermax=max_iter)
        bary = np.reshape(bary, (x_size,y_size)) 
        
        nanmin = np.nanmin(bary)
        nanmax = np.nanmax(bary)
#Finding max and min intensities for consistent plotting
        #Finding the Min intensity with NAN handler
        #print(vmin)
        #print(nanmin)
        if vmin == []:
            if np.isnan(nanmin) == True:
                vmin = []
            else:
                vmin = nanmin
        ##If NAN, do nothing
        ##If min(bary) > vmin, do nothing
            #print(vmin)
        else:
            if np.isnan(nanmin) == False:
                barymin = nanmin
                if  barymin < vmin:
                    vmin = barymin 
            #print(vmin)
        if np.isnan(vmin):
            vmin = 0
        #print(vmin)
        #Finding the Max intensity with NAN handler
        if vmax == []:
            if np.isnan(nanmax) == True:
                vmax = []
            else:
                vmax = nanmax
        ##If NAN, do nothing
        ##If max(bary) < vmax, do nothing
        else:
            if np.isnan(nanmax) == False:
                barymax = nanmax
                if  barymax > vmax:
                    vmax = barymax     
        if np.isnan(vmax):
            vmax = 1

            
            
                  
        title = "bary" + file[15:-4] 
        params = "_reg_" + str(reg) + "_c_" + str(c) + "_iters_" + str(max_iter) + "_intensity_" + str(intensity) 
        if save:
            np.save("./results/kbcm_bary/" + title + params + ".npy", bary)
        
        
        
##Plotting set up for either 4 or 6 groups of noise levels
#Choose the intensity scale on the plot to be eith 0/1 or min/max 
    if plot:
        k = 1
        params = "_reg_" + str(reg) + "_c_" + str(c) + "_iters_" + str(max_iter) + "_intensity_" + str(intensity)

        noise_lvls = ["0.000", "0.100", "0.200", "0.500", "1.000"]
        m = 3
        for n in noise_lvls: 
            barys = np.load("./results/kbcm_bary/bary_noiselvl_" + n + params + ".npy")
            plt.subplot(2, m, k)
            plt.title("bary_lvl_"+n+"_mean_0.000")
            ##added vmin and vmax so all plots have same itensity scale
            k += 1
            if intensity == "zeroone":
                plt.imshow(barys, vmin=0, vmax=1)
            elif intensity == "minmax":
                plt.imshow(barys, vmin=vmin, vmax=vmax)
            else:
                plt.imshow(barys)
        

   

    if save:
        plt.savefig("./results/kbcm_bary/plots_kbcm_bary/kbcm_" + title + params + ".png")    
   
    if plot:
        plt.show()



if __name__ == "__main__":
    reg = [.25]
    for r in reg:
        kbcm_bary(reg = r, c = -.7, max_iter = 10, save=True, intensity = "maxmin")
  
    

"""
Notes for Richard

KBCM:
#reg = [.001, .01] (take the longest to run, couldn’t get them to work)

#was able to run these easily/quickly but should rerun with the new data set 
#and with more iterations
#reg = .1 with max_iter=100     c =-.5
#reg = [.05,.5,.9] with max_iter=100  c =-.5    
#reg = [.25,.4,.6] with max_iter=100  c =-.5  

#reg = [.025, .05, .075, .1, .3, .5, .7, .9]
#max_iter = [100, 500, 1000, 2500, 5000] (I found interesting results at low iterations [10] but not sure if this is worth it)
#cs = -[.001, .01, .1, .5, .7, .9] (haven’t experimented with a lot of c, but could be worth it to see if the step matters [note it’s negative])

"""

 


     
"""  
#Notes for me to remember
#run overnight       
#reg = .001 is taking a very long time to run
#reg = .01 is taking along time to run        

#already finished
#reg = .1 with max_iter=100     c =-.5
#reg = [.05,.5,.9] with max_iter=100  c =-.5    
#reg = [.25,.4,.6] with max_iter=100  c =-.5   

     
if __name__ == "__main__":
    reg = [.001, .05, .1, .5, .9]
    max_iter = [100, 500, 750, 1000, 5000]
    cs = [.001, .01, .1, .5, .7, .9]
    for m in max_iter:
        for r in reg:
            for c in cs:
                kbcm_bary(reg = r, c=-c, max_iter = m)

#reg = [.001, .025, .05, .075, .1, .15, .2, .3, .5, .7, .9]
#max_iter = [100, 500, 750, 1000, 2500, 5000]
#cs = [.001, .01, .1, .5, .7, .9]

"""
    
