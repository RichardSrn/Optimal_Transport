#! /usr/bin/env python3

import sys
import numpy as np

from const import DATA_PATH, TEST_ALG, PLOT_TITLE
from hist_from_images import hist_from_images
from coupling_from_2_hist import coupling_from_2_hist
from barycenter_from_coupling import barycenter_from_coupling
from plot_baryc import plot_baryc

def main(plot = False, plot_title = None):
    #define the rng
    rng = np.random.RandomState(42)

    #choose 2 images
    index = rng.choice(np.arange(199), 2, replace=False)
    img1, img2 = np.load(DATA_PATH)[index]

    #make it absolute value
    img1 = abs(img1)
    img2 = abs(img2)

    #normalize the data 
    img1 = img1 / img1.sum()
    img2 = img2 / img2.sum()

    #turn 2D images into 1D vector --histogram--
    hist1, hist2 = hist_from_images(img1, img2)

    #get the coupling matrix
    coupling = coupling_from_2_hist(hist1, hist2, TEST_ALG)

    #get the barycenter
    barycenter = barycenter_from_coupling(hist1, hist2, coupling)

    if plot :
        if plot_title is None :
            plot_baryc(img1, img2, barycenter, title = PLOT_TITLE)
        else :
            plot_baryc(img1, img2, barycenter, title = plot_title)

    return barycenter


if __name__ == "__main__" :
    """
    Use const.py to change constants.
        - **DATA_PATH** is the path to the data to use. Must be a np.ndarray format.
        - **TEST_ALG** is the algorithm to test. 
                The algorithm must take as input two histograms and the cost matrix.
                It must output a coupling matrix.
        - **PLOT_TITLE** is the title of the final plot.
    
    Plot the results :
        To plot the results you must run the *main.py* as *./main.py plot* 
            in which case the plot will take the name of the tested algorithm.
        Otherwise you can run run *./main.py "desired_plot_title"*. 
    """
    if len(sys.argv) == 1 :
        main()
    else :
        if sys.argv[1] == "plot" :
            main(plot = True)
        else :
            main(plot =True, plot_title = sys.argv[1])

