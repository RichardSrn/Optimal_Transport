#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import torch

import sinkhorn_barycenters as sink
from computeK import computeK

"""
intensity = "minmax" or "zeroone"
    Will set the plots based off of the minimum and maximum intensity values across all noise levels
    Or sets the plot's intensity to be between 0 and 1
    In case of NAN values? Set to 0 and 1
    
imgs: number of images used to compute the barycenter

noise_lvl: choosing to use 4 different or 6 different noise levels for barycenters and plots
"""


def get_files(path="./data"):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]
    onlyfiles.sort()

    return onlyfiles

def check_in_files(existing_files, cheking_file):
    if any(cheking_file in existing_files) :
        print(f"{cheking_file} already exists.")
        return True
    return False


def debiased_sink_bary(epsilon=.1, max_iter=int(1000), intensity="zeroone", plot=True, save=True):
    files = get_files()

    # if plot:
    #     plt.figure(1, figsize=(15, 10))
    # vmin = []
    # vmax = []

    existing_files = get_files("./results/debiased_sink_bary")

    for file in files:
        data = np.load("./data/" + file)
        data = abs((data[:]))
        title = "bary" + file[15:-4]
        if epsilon == 1e-5 :
            params = "_eps_0.00001" + "_iter_" + str(max_iter) + "_intensity_" + str(intensity)
        else :
            params = "_eps_" + str(epsilon) + "_iter_" + str(max_iter) + "_intensity_" + str(intensity)

        if check_in_files(existing_files, title + params + ".npy") :
            bary = np.load("./results/debiased_sink_bary/" + title + params + ".npy")
        else :
            # data_pos = data - np.min(data)
            # mass = np.sum(data_pos, axis=0).max()
            # unbalanced data
            # hs = data_pos / mass
            # normalized data
            # mass_hs = np.sum(hs, axis=0)
            # hs_hat = hs / mass_hs


            # normalized
            data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
            print(np.nanmin(data), np.nanmax(data), )

            # Computing barycenter
            """using hs for unbalances or hs_hat for normalized"""
            P, K = computeK(data, epsilon)
            bary = sink.barycenter(P, K, reference="debiased", maxiter=max_iter)
            print(np.nanmin(bary), np.nanmax(bary))
            np.save("./results/debiased_sink_bary/" + title + params + ".npy", bary)

        # Finding max and min intensities for consistent plotting
        # Finding the Min intensitywith NAN handler
        # if vmin == []:
        #     if torch.isnan(torch.min(bary)) == True:
        #         vmin = []
        #     else:
        #         vmin = torch.min(bary)
        #         vmin = vmin.numpy()
        ##If NAN, do nothing
        ##If min(bary) > vmin, do nothing
        # else:
        #     if torch.isnan(torch.min(bary)) == False:
        #         barymin = torch.min(bary)
        #         barymin = barymin.numpy()
        #         if barymin < vmin:
        #             vmin = barymin
        # if np.isnan(vmin):
        #     vmin = 0

        # Finding the Max intensity with NAN handler
        # if vmax == []:
        #     if torch.isnan(torch.max(bary)) == True:
        #         vmax = []
        #     else:
        #         vmax = torch.max(bary)
        #         vmax = vmax.numpy()
        ##If NAN, do nothing
        ##If max(bary) < vmax, do nothing
        # else:
        #     if torch.isnan(torch.max(bary)) == False:
        #         barymax = torch.max(bary)
        #         barymax = barymax.numpy()
        #         if barymax > vmax:
        #             vmax = barymax
        # if np.isnan(vmax):
        #     vmax = 1
        # print(vmax)
        # saving the dataset
        
        # print(vmax)

    # # Choose the intensity scale on the plot to be eith 0/1 or min/max
    # if plot:
    #     k = 1
    #     params = "_eps_" + str(epsilon) + "_iter_" + str(max_iter) + "_intensity_" + str(intensity)
    #     # if noise_lvl == 6:
    #     noise_lvls = ["0.000", "0.100", "0.200", "0.500", "1.000"]
    #     m = 3
    #     for n in noise_lvls:
    #         barys = np.load("./results/debiased_sink_bary/bary_noiselvl_" + n + params + ".npy")
    #         plt.subplot(2, m, k)
    #         plt.title("bary_lvl_" + n + "_mean_0.000")
    #         ##added vmin and vmax so all plots have same itensity scale
    #         k += 1
    #         if intensity == "zeroone":
    #             plt.imshow(barys, vmin=0, vmax=1)
    #         elif intensity == "minmax":
    #             plt.imshow(barys, vmin=vmin, vmax=vmax)
    #         else:
    #             plt.imshow(barys)

    # if save:
    #     plt.savefig("./results/debiased_sink_bary/plots_debiased_sink_bary/" + title + params + ".png")

    # if plot:
    #     plt.show()


if __name__ == "__main__":
    # iters = [100, 750, 2000, 10000, 1e8]
    # for i in iters:
    debiased_sink_bary(epsilon=.3, max_iter=100, intensity="minmax")

"""Notes for Richard
Debiased:
#epsilon = .001 max_iter = int(1e6)
#epsilon = .005 max_iter = int(1e6)
#epsilon = .01, max_iter = int(1e6), int(1e8) (only works for greater than 1e5)
#epsilon = .05, max_iter = int(1e7) (anything less than 1e5 doesn't work)
#epsilon = .2   max_iter = [2500, 3000, 4000]
#eps = .5 and iters = [500, 750, 1000]
#eps = .7 and iters = [100, 750, 1000]   
"""

# Notes for me to remember
# takes a long time to run
# epsilon = .0001, max_iter = int(1e6)
# epsilon = .05, max_iter = int(1e7)

# eps = [.0001, .005, .01, .05, .1, .2, .5, .7]
# iters = [100, 750, 2000, 1e8]

# [.001, .025, .05, .075, .1, .15, .2, .3, .5, .7, .9]


# ALREADY RUN
# eps = .05 and iters = [ ] (try with high iterations like 1e6,8,10  [[100, 500, 1000, 2000, 3000, 5000, 1e5] don't work
# eps = .1 and iters = [100, 10000, ] Need more than 10000 for .1
# eps = .2 and iters = [2500, 5000] lower than 2500 is no good, 5000 is no good
# eps = .5 and iters = [500, 750, 1000]
# eps = .7 and iters = [100, 750, 1000]
