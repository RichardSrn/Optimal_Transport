#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np

from barycenter_model import tlp_bi


# parameters
#    reg : float
#        Entropic regularization term
#    eta : float
#        The parameter for cost matrix
#    outItermax: int
#        Max number of iterations for outer loop
#    inItermax: int
#        Max number of iterations for inner loop
#    outstopThr : float
#        Stop threshold for outer loop
#    instopThr : float
#        Stop threshold for inner loop 
###


def get_files():
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]

    onlyfiles.sort()

    for file in onlyfiles:
        yield file


def tlp_bary(reg=0.1, 
             eta=0.1, 
             x_size=50, 
             y_size=50, 
             outItermax=10, 
             weights=None, 
             inItermax=100, 
             outstopThr=1e-8,
             instopThr=1e-8, 
             log=False, 
             intensity="zeroone", 
             samples=200, 
             plot=True, 
             save=True):
    print(f"START tlp_bary\n\
          reg={reg},eta={eta},outItermax={outItermax},inItermax={inItermax},\
          outstopThr={outstopThr},instopThr={instopThr},intensity={intensity},samples={samples}")

    files = get_files()
    # if plot:
    #     plt.figure(1, figsize=(15, 10))
    #     k = 1
    #     vmin = []
    #     vmax = []

    for file in files:
        print(f"Current file is : {file}")

        data = np.load("./data/" + file)
        data = abs(data[:samples])  ##number of images to use to compute barycenter

        data = np.reshape(data, (len(data), (x_size * y_size)))
        # data = data.reshape((-1, x_size * y_size))
        # Computing barycenter
        data = data.T
        data_pos = data - np.min(data)
        mass = np.sum(data_pos, axis=0).max()
        # unbalanced data
        hs = data_pos / mass
        # normalized data
        mass_hs = np.sum(hs, axis=0)
        hs_hat = hs / mass_hs

        # barycenter of tlp_bi
        bary, barys = tlp_bi(hs, 
                             hs_hat, 
                             x_size, 
                             y_size, 
                             reg, 
                             eta, 
                             weights, 
                             outItermax,
                             inItermax, 
                             outstopThr,
                             instopThr, 
                             log=log)
        # print(bary[0])
        bary = np.reshape(bary, (x_size, y_size))
        # print(bary[0])
        title = "bary" + file[15:-4]
        params = "_reg_" + str(reg) + "_eta_" + str(eta) + "_outer-inner_" + str(outItermax) + "-" + str(inItermax) + "_samples_" + str(samples) + "_intensity_" + str(intensity)
        np.save("./results/tlp_bary/" + title + params + ".npy", bary)
        print("SAVED -- "+"./results/tlp_bary/" + title + params + ".npy")


    ### PLOTTING ###
        # nanmin = np.nanmin(bary)
        # nanmax = np.nanmax(bary)
        # # Finding max and min intensities for consistent plotting
        # # Finding the Min intensity with NAN handler
        # # print(vmin)
        # # print(nanmin)
        # if vmin == []:
        #     if np.isnan(nanmin) == True:
        #         vmin = []
        #     else:
        #         vmin = nanmin
        #         # vmin = vmin.numpy()
        #     ##If NAN, do nothing
        #     ##If min(bary) > vmin, do nothing
        #     print(vmin)
        # else:
        #     if np.isnan(nanmin) == False:
        #         barymin = nanmin
        #         # barymin = barymin.numpy()
        #         if barymin < vmin:
        #             vmin = barymin
        #     print(vmin)
        # if np.isnan(vmin):
        #     vmin = 0
        # print(vmin)
        # # Finding the Max intensity with NAN handler
        # if vmax == []:
        #     if np.isnan(nanmax) == True:
        #         vmax = []
        #     else:
        #         vmax = nanmax
        #         # vmax = vmax.numpy()
        # ##If NAN, do nothing
        # ##If max(bary) < vmax, do nothing
        # else:
        #     if np.isnan(nanmax) == False:
        #         barymax = nanmax
        #         # barymax = barymax.numpy()
        #         if barymax > vmax:
        #             vmax = barymax
        # if np.isnan(vmax):
        #     vmax = 1


    ##Plotting set up for either 4 or 6 groups of noise levels
    # Choose the intensity scale on the plot to be eith 0/1 or min/max
    # if plot:
    #     k = 1
    #     noise_lvls = ["0.000", "0.100", "0.200", "0.500", "1.000"]
    #     m = 3
    #     for n in noise_lvls:
    #         barys = np.load("./results/tlp_bary/bary_noiselvl_" + n + params + ".npy")
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
    #     plt.savefig("./results/tlp_bary/tlp_" + title + params + ".png")

    # if plot:
    #     plt.show()
    print(f"END tlp_bary\n\
          reg={reg},eta={eta},outItermax={outItermax},inItermax={inItermax},\
          outstopThr={outstopThr},instopThr={instopThr},intensity={intensity},samples={samples}")


if __name__ == "__main__":
    # reg = [.001, .01, .05, .1, .5, .9]
    # eta = [.001, .1]#, .05, .7]
    # for r in reg:
    #    for e in eta:
    #        tlp_bary(reg = r, eta = e)#, outItermax = 20, inItermax=100, outstopThr=1e-10, instopThr=1e-10)

    tlp_bary(reg=.05, eta=.1,
             intensity="minmax")  # , outItermax = 20, inItermax=100, outstopThr=1e-10, instopThr=1e-10)

#    reg = [.05] #[.01, .05, .1, .5]
#    for r in reg:
#        tlp_bary(reg = r)
