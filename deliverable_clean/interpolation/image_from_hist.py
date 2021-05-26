#! /usr/bin/env python3
import numpy as np

def image_from_hist(hist : np.ndarray, size_x, size_y):
    image = hist.reshape((size_x,size_y))
    return image