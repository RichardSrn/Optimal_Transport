#! /usr/bin/env python3

"""
Then, we use several criteria to characterize the properties 
of the barycenters: the maximum amplitude of the image, the 
number of voxels above a threshold defined as half of the 
maximum amplitude – which provides an estimate of the size 
of the most active regions, and standard deviation of the 
coordinates of all above-threshold voxels – which measures 
the spatial spread of activated regions.
"""


import numpy as np


def max_amplitude(image) :
    """
    Get the max amplitude of the image.
    """
    
    return np.max(image)

def pixel_above_thr(image, threshold) :
    """
    Get the pixels' coordinates which have amplitude 
    greater then or equal to half the maximum amplitude.
    """

    return np.array(np.where(image >= threshold / 2))

def sd_pixels(voxels) :
    """
    Get the standard deviation of the above-threshold pixels coordinates.

    sigma = sqrt(sum_{i=0}^n( (xi – mu)2 / n ))
    """

    mu = np.array([np.mean(voxels, axis=1)]).T
    
    sigma = np.array([np.sqrt(np.sum( np.square(voxels - mu), axis=1 ) / voxels.shape[0]  )]).T
    return sigma


def criteria(image) :
    """
    Returns the 3 criteria :
    - max amplitude
    - coordinate of above-threshold pixels --threshold = max_amplitude/2--
    - standard deviation of the above-threshold pixels.
    """
    print(image)
    if any(image == None):
        return (None,None,None)
    max_ampl = max_amplitude(image)
    pixels = pixel_above_thr(image, max_ampl)
    std = sd_pixels(pixels)
    return (max_ampl, pixels, std)


if __name__ == '__main__':
    image = np.array([[0,1,.5],
                      [1,0,1.5],
                      [.3,1,0]])
    criteria(image)
    