#! /usr/bin/env/ python3

import numpy as np
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve
from gilbert2d import hilbert2d

def vanilla_space_filling_curve(shape):
    # dist = image.reshape((np.prod(image.shape, -1)))

    nx, ny = shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xv, yv = np.meshgrid(y, x)
    vanilla_curve = np.vstack((xv.flatten(), yv.flatten()))
    return vanilla_curve

def hilbert_space_filling_curve(shape):
    hilbert_curve = np.array([[x,y] for x,y in hilbert2d(shape[0], shape[1])]).T
    return hilbert_curve

def hist_from_images(image,curve):
    return image[tuple(curve)]


def plot(img, curve1, hist1, curve2, hist2):
    plt.figure(1, figsize=(10,10))

    plt.subplot(2,2,1)
    plt.title("2D-image - vanilla_curve")
    plt.imshow(img)
    plt.plot(curve1[0], curve1[1], alpha=.5, color = "red")
    # plt.axis("off")

    plt.subplot(2,2,2)
    plt.title("1D-intensity distribution - vanilla_curve")
    plt.bar(np.arange(2500),hist1, color = "red", linewidth=1)
    # plt.axis("off")

    plt.subplot(2,2,3)
    plt.title("2D-image - hilbert_curve")
    plt.imshow(img)
    plt.plot(curve2[0], curve2[1], alpha=.5, color = "green")
    # plt.axis("off")

    plt.subplot(2,2,4)
    plt.title("1D-intensity distribution - hilbert_curve")
    plt.bar(np.arange(2500),hist2, color = "green", linewidth=1)
    # plt.axis("off")

    plt.show()



def main():
    image = np.load("./artificial_data.npy")[1]

    vanilla_curve = vanilla_space_filling_curve(image.shape)
    vanilla_hist = hist_from_images(image, vanilla_curve)
    
    hilbert_curve = hilbert_space_filling_curve(image.shape)
    hilbert_hist = hist_from_images(image, hilbert_curve)
    plot(image, vanilla_curve, vanilla_hist, hilbert_curve, hilbert_hist)

if __name__ == "__main__":
    main()