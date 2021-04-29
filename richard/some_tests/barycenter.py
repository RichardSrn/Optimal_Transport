import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import ot
import pylab as pl
from ot.datasets import make_1D_gauss as gauss
from ot.datasets import make_2D_samples_gauss as ggauss


def timer(func):
    # compute execution time
    def inner(*args, **kwargs):
        print(f"\n\nExecution of {func.__name__!r}")
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'DONE. {func.__name__!r}.\n\tElapsed time {(t2 - t1):.3f}s')
        return result

    return inner


@timer
def get_images(name1, name2, name3, name4):
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    print("\tGetting the images in the directory :", CURR_DIR + "/data/")
    f1 = 1 - pl.imread(f"{CURR_DIR}/data/" + name1).astype(np.float64)[:, :, 2]
    f2 = 1 - pl.imread(f"{CURR_DIR}/data/" + name2).astype(np.float64)[:, :, 2]
    f3 = 1 - pl.imread(f"{CURR_DIR}/data/" + name3).astype(np.float64)[:, :, 2]
    f4 = 1 - pl.imread(f"{CURR_DIR}/data/" + name4).astype(np.float64)[:, :, 2]
    return f1, f2, f3, f4


@timer
def resize_images(f1, f2, f3, f4, size):
    print("\tImages original shape :", f1.shape)
    step_x = f1.shape[0] // size[0]
    step_y = f1.shape[1] // size[1]
    f1 = f1[1::step_x, 1::step_y]
    f2 = f2[1::step_x, 1::step_y]
    f3 = f3[1::step_x, 1::step_y]
    f4 = f4[1::step_x, 1::step_y]
    print("\tImages are now shaped as :", f1.shape)
    return f1, f2, f3, f4


@timer
def stack_images(*args):
    A = []
    for f in args:
        f = f / np.sum(f)
        A.append(f)
    A = np.array(A)
    return A


@timer
def make_interp_imgs(A, nb_images, reg=0.004):
    # those are the four corners coordinates that will be interpolated by bilinear interpolation
    v1 = np.array((1, 0, 0, 0))
    v2 = np.array((0, 1, 0, 0))
    v3 = np.array((0, 0, 1, 0))
    v4 = np.array((0, 0, 0, 1))

    for i in range(nb_images):
        for j in range(nb_images):
            if i == j == 0 or \
                    i == j == nb_images - 1 or \
                    (i == 0 and j == nb_images - 1) or \
                    (i == nb_images - 1 and j == 0):
                continue
            tx = float(i) / (nb_images - 1)
            ty = float(j) / (nb_images - 1)
            # weights are constructed by bilinear interpolation
            tmp1 = (1 - tx) * v1 + tx * v2
            tmp2 = (1 - tx) * v3 + tx * v4
            weights = (1 - ty) * tmp1 + ty * tmp2
            yield ot.bregman.convolutional_barycenter2d(A, reg, weights)


@timer
def make_plot(f1, f2, f3, f4, interp_images, nb_images, reg=0.004):
    pl.figure(figsize=(10, 10))
    cm = 'Blues'

    pl.suptitle('Convolutional Wasserstein Barycenters in POT\n' + str(nb_images) +
                ' interpolation images - regularization = ' + str(round(reg, 10)),
                fontsize=20,
                fontweight="bold")

    for i in range(nb_images):
        for j in range(nb_images):
            pl.subplot(nb_images, nb_images, i * nb_images + j + 1)

            if i == 0 and j == 0:
                pl.imshow(f1, cmap=cm)
                pl.axis('off')
            elif i == 0 and j == (nb_images - 1):
                pl.imshow(f3, cmap=cm)
                pl.axis('off')
            elif i == (nb_images - 1) and j == 0:
                pl.imshow(f2, cmap=cm)
                pl.axis('off')
            elif i == (nb_images - 1) and j == (nb_images - 1):
                pl.imshow(f4, cmap=cm)
                pl.axis('off')
            else:
                # call to barycenter computation
                pl.imshow(next(interp_images), cmap=cm)
                pl.axis('off')
    pl.show()


@timer
def barycenter(size=(50, 50), reg=0.004):
    f1, f2, f3, f4 = get_images("wilber.png", "rhino.png", "owl.png", "dog.png")
    f1, f2, f3, f4 = resize_images(f1, f2, f3, f4, size=size)
    A = stack_images(f1, f2, f3, f4)
    nb_images = 5
    interp_images = make_interp_imgs(A, nb_images, reg=reg)
    make_plot(f1, f2, f3, f4, interp_images, nb_images, reg=reg)


@timer
def barycenter2():
    n = 100
    x = np.arange(n, dtype=np.float64).reshape((n, 1))

    dist1 = gauss(n, m=20, s=5)
    dist2 = gauss(n, m=60, s=10)

    # loss matrix + normalization
    m = ot.dist(x, x)
    m /= m.max()
    a = stack_images(dist1, dist2).T
    n_distributions = a.shape[1]

    # Regularization tuning
    ## computing
    regs = [1.0, 0.1, 0.01, 0.001]
    reg_tun = []
    for reg in regs:
        reg_tun.append(ot.barycenter(a, m, reg=reg))

    ## plotting
    pl.figure(1, figsize=(10, 12))
    pl.subplot(3, 1, 1)
    for i in range(n_distributions):
        pl.plot(x, a[:, i], label="original distr.", linewidth=3, color="black", alpha=1 / (2 + i * 2))
    i = 0
    for y, reg in zip(reg_tun, regs):
        pl.plot(x, y, label="reg=" + str(reg), color="red", alpha=1 / (1 + i))
        i += 1
    pl.title("Regularization tuning. Unweighted barycenter.")
    pl.legend(loc="best")

    # weight tuning - l2 norm
    ## computing
    weights = [.1, .3, .5, .7, .9]
    weig_tun_l2 = []
    for w in weights:
        weig = np.array([1 - w, w])
        weig_tun_l2.append(a.dot(weig))
    ## plotting
    pl.subplot(3, 1, 2)
    for i in range(n_distributions):
        pl.plot(x, a[:, i], label="original distr.", linewidth=3, color="black", alpha=1 / (2 + i * 2))
    i = 0
    colors = ["red", "magenta", "blue", "cyan", "green", "yellow"]
    for y, weig in zip(weig_tun_l2, weights):
        pl.plot(x, y, label="weights=" + str(weig), color=colors[i])
        i += 1
    pl.title("Weight tuning. l2 norm.")
    pl.legend(loc="best")

    # weight tuning - Wasserstein Distance
    ## computing
    weights = [.1, .3, .5, .7, .9]
    weig_tun_w = []
    for w in weights:
        weig = np.array([1 - w, w])
        weig_tun_w.append(ot.barycenter(a, m, reg=0.001, weights=weig))
    ## plotting
    pl.subplot(3, 1, 3)
    for i in range(n_distributions):
        pl.plot(x, a[:, i], label="original distr.", linewidth=3, color="black", alpha=1 / (2 + i * 2))
    i = 0
    for y, weig in zip(weig_tun_w, weights):
        i += 1
        pl.plot(x, y, label="weights=" + str(weig), color="blue", alpha=1 / (1 + i))
    pl.title("Weight tuning. Wasserstein distance - Regularization = 0.001.")
    pl.legend(loc="best")

    pl.suptitle("Barycenter \n l2-norm and entropic regularization.", fontsize=14)
    pl.show()


@timer
def barycenter3(save=False):
    if save:
        # Get current directory
        CURR_DIR = os.path.dirname(os.path.realpath(__file__))

    # Initial parameters
    rng = np.random.RandomState(4269)
    n_pt = 10
    n_dist = 3

    # Generate the distributions
    dists = []
    for i in range(n_dist):
        mu = (rng.random(size=(1, 2)) - 0.5) * 40 + 50
        sigma_xy = rng.random(size=(2)) * 5 + 4
        rho = rng.randint(0, 1)
        sigma = np.array([[sigma_xy[0] ** 2, np.prod(sigma_xy) * rho],
                          [np.prod(sigma_xy) * rho, sigma_xy[1] ** 2]])
        dists.append(ggauss(n_pt, m=mu, sigma=sigma, random_state=42))
    dists = np.array(dists)

    k = 0
    stop = 3
    step = 0.3
    x = np.arange(start=0, stop=stop, step=step)
    y = np.exp(-x ** 2)

    plt.scatter(x,y,color="blue",alpha=.7, label="f(x)=exp(-x^2)")
    plt.xlabel("x in 0 to 3 step 0.03")
    plt.ylabel("Regularization input")
    plt.legend()
    plt.title("Input for barycenter's regularization parameter.")
    plt.show()

    # Plot the distributions alone (no barycenter)
    plt.figure(figsize=(5, 5))
    plt.xlim(10, 90)
    plt.ylim(10, 90)
    for i in range(n_dist):
        plt.scatter(dists[i, :, 0], dists[i, :, 1], alpha=.5)
    plt.title(str(n_dist)+" bivariate normal distributions.")
    plt.show()

    for reg in y:
        k += 1
        print("graph", k, "over", stop // step)
        # Compute the regularized (entropy) barycenter
        d = ot.bregman.convolutional_barycenter2d(dists, reg=reg)

        # Plotting
        ## Plot the distributions
        plt.figure(figsize=(5, 5))
        # plt.xlim(10, 90)
        # plt.ylim(10, 90)
        for i in range(n_dist):
            plt.scatter(dists[i, :, 0], dists[i, :, 1], alpha=.5)

        ## Plot the barycenter
        plt.scatter(d[:, 0], d[:, 1], color="black", s=30, alpha=.7, label="barycenter")
        plt.title("Bregman convolutional barycenter\nRegularization = " + str(round(reg, 5)), fontsize=16)
        pl.legend(loc="upper right")
        if save:
            plt.savefig(CURR_DIR + "/results/barycenter/" + str(k) + ".jpg")
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        barycenter()
    else:
        size = (int(sys.argv[1]), int(sys.argv[2]))
        if len(sys.argv) > 3:
            reg = int(sys.argv[3])
            barycenter(size=size, reg=reg)
        else:
            barycenter(size=size)
