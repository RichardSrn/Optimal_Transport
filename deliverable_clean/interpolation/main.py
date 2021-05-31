#! /usr/bin/env python3

"""
Use const.py to change constants.
    - **DATA_PATH** is the path to the data to use. Must be a np.ndarray format.
    - **TEST_ALG** is the algorithm to test. 
            The algorithm must take as input two histograms and the cost matrix.
            It must output a coupling matrix.
    - **PLOT_TITLE** is the title of the final plot.


Execute script and behavior :
    - *python3 main.py* will run the code and save the barycenter's matrix.py
    - *python3 main.py <paramters>* :
        . if *show* is in parameters : **show** the plot.
        . if *save* is in parameters : **save** the plot.
        . anything else is used as **plot_title**.
            . if only a title is passed as parameter, save the plot with this title.
        . if no title specified, the algorithm to test name is used as title.

    -> example : "python3 main.py test_title show save"
                            will show and save the plot with "test_title" as title.
"""

import sys
from time import time

import numpy as np

from barycenter_from_coupling import barycenter_from_coupling
from const import DATA_PATH, TEST_ALG, PLOT_TITLE
from coupling_from_2_hist import coupling_from_2_hist
from hist_from_images import hist_from_images
from image_from_hist import image_from_hist
from plot_baryc import plot_baryc


def main(img1=None, img2=None, save=False, show=False, plot_title=None, seed=42, absolute=True):
    t = time()
    # define the rng
    rng = np.random.RandomState(seed)

    if img1 is None:
        # choose 2 images
        print("\nNo image input, choose 2 random images from \n", DATA_PATH, "\nRNG_seed = ", seed, sep="")
        index = rng.choice(np.arange(199), 2, replace=False)
        img1, img2 = np.load(DATA_PATH)[index]
        print("DONE.")

    # get image shape, assuming both images are same shape.
    size_x, size_y = img1.shape

    if absolute:
        print("\nTurn images' values absolute to avoid errors.")
        # make it absolute value
        img1 = abs(img1)
        img2 = abs(img2)
        print("DONE. t =", round(time() - t, 2), "s.")

        print("\nNormalize the data.")
        # normalize the data
        img1 = img1 / img1.sum()
        img2 = img2 / img2.sum()
        print("DONE. t =", round(time() - t, 2), "s.")

    else:
        print("\nNormalize the data.")
        # normalize the data
        img1 = img1 / abs(img1).sum()
        img2 = img2 / abs(img2).sum()
        print("DONE. t =", round(time() - t, 2), "s.")

    print("\nReshape the 2D images into 1D histograms.")
    # turn 2D images into 1D vector --histogram--
    hist1, hist2 = hist_from_images(img1, img2)
    print("DONE. t =", round(time() - t, 2), "s.")

    print("\nGet coupling matrix from the two histograms;")
    print("use", TEST_ALG, ".")
    # get the coupling matrix
    coupling = coupling_from_2_hist(hist1, hist2, TEST_ALG, size_x, size_y)
    print("DONE. t =", round(time() - t, 2), "s.")

    print("\nGet histogram barycenter from coupling matrix.")
    # get the barycenter
    # hist_barycenter = barycenter_from_coupling(coupling, size_x, size_y)
    barycenter = barycenter_from_coupling(coupling, size_x, size_y)
    print("DONE. t =", round(time() - t, 2), "s.")

    # print("\nReshape histogram barycenter into 2D barycenter.")
    # # turn 1D histogram into 2D image
    # barycenter = image_from_hist(hist_barycenter, size_x, size_y)
    # print("DONE. t =", round(time() - t, 2), "s.")

    if show or save:
        print("\nMake plot.")
        if plot_title is None:
            plot_baryc(img1, img2, barycenter, t=t, title=PLOT_TITLE, show=show, save=save)
        else:
            plot_baryc(img1, img2, barycenter, t=t, title=plot_title, show=show, save=save)
        print("DONE. t =", round(time() - t, 2), "s.")

    print("\nSave barycenter matrix as barycenter.npy.")
    np.save("barycenter.npy", barycenter)
    print("DONE. t =", round(time() - t, 2), "s.")

    return barycenter


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        show = False
        save = True
        parameters = sys.argv[1:]
        if "show" in parameters:
            idx = parameters.index("show")
            parameters.pop(idx)
            show = True
            save = False
        if "save" in parameters:
            idx = parameters.index("save")
            parameters.pop(idx)
            save = True
        plot_title = " ".join(parameters)
        if plot_title != "":
            main(show=show, save=save, plot_title=plot_title)
        else:
            main(show=show, save=save)
