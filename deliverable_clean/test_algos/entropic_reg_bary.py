#! /usr/bin/env python3
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import ot


def get_files():
    onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
    onlyfiles = [file for file in onlyfiles if file[-5:] == "0.npy"]
    onlyfiles.sort()

    for file in onlyfiles:
        yield file


def entropic_reg_bary(reg = 0.04, plot=False, save=False, show=False):
    files = get_files()

    if plot:
        plt.figure(1, figsize=(15, 10))
        k = 1

    for file in files:
        title = "bary" + file[15:-4] + "_reg_" + str(reg)
        data = np.load("./data/" + file)
        bary = ot.bregman.convolutional_barycenter2d(data, reg=reg)
        np.save("./results/entropic_reg_bary/" + title + ".npy", bary)

        if plot:
            plt.subplot(2, 3, k)
            plt.title(title[:len(title)//2]+"\n"+title[len(title)//2:])
            plt.imshow(bary)
            k += 1

    if show:
        plt.show()
    if save:
        plt.savefig("./results/entropic_reg_bary/entropic_"+str(reg)+"_reg_"+str(data.shape[0])+"_samples.png")


if __name__ == "__main__":
    for reg in [0.4,4] :
        entropic_reg_bary(reg=reg)
