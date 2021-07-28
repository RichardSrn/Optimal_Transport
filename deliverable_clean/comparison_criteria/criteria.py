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


def max_amplitude(image):
    """
    Get the max amplitude of the image.
    """
    maximum_amplitude = np.max(image)-np.min(image)
    location = np.where(image == maximum_amplitude+np.min(image))

    return (maximum_amplitude, location)


def pixel_above_thr(image, threshold):
    """
    Get the pixels' coordinates which have amplitude 
    greater then or equal to half the maximum amplitude.
    """

    return (np.array(np.where(image >= threshold / 2)), image[np.where(image >= threshold / 2)])


def std_pixels(pixels):
    """
    Get the standard deviation of the above-threshold pixels coordinates.

    sigma = sqrt(sum_{i=0}^n( (xi – mu)2 / n ))
    """

    mu = np.array([np.mean(pixels, axis=1)]).T

    sigma = np.sqrt(np.sum(np.square(pixels - mu), axis=1) / pixels.shape[0])
    return sigma


def get_barycenter_location(loc_pixels_abv_thld, value_pixels_abv_thld):
    if len(value_pixels_abv_thld) != 0:
        value_pixels_abv_thld /= value_pixels_abv_thld.sum()
        average = np.zeros(shape=(1, 2))
        for i in range(len(value_pixels_abv_thld)):
            average += loc_pixels_abv_thld[:, i] * value_pixels_abv_thld[i]
        return average[0]
    else:
        return np.array([np.inf, np.inf])


def criteria(image):
    """
    Returns the 3 criteria :
    - max amplitude
    - coordinate of above-threshold pixels --threshold = max_amplitude/2--
    - standard deviation of the above-threshold pixels.
    """
    if np.isnan(image).all():
        return (None, None, None, None)
    max_ampl, max_ampl_loc = max_amplitude(image)
    if max_ampl == 0:
        return (None, None, None, None)
    loc_pixels_abv_thld, value_pixels_abv_thld = pixel_above_thr(image, max_ampl)
    nb_pixels_abv_thld = len(loc_pixels_abv_thld[0])
    std = std_pixels(loc_pixels_abv_thld)
    loc_barycenter = get_barycenter_location(loc_pixels_abv_thld, value_pixels_abv_thld)
    return (max_ampl, loc_barycenter, nb_pixels_abv_thld, std)


if __name__ == '__main__':
    image = np.array([[0, 1, .5],
                      [1.5, 2, 1.5],
                      [.3, 1, 0]])
    criteria(image)
