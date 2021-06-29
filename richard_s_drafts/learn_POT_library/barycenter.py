import math
import os
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
    cm = 'Greys'

    pl.suptitle('Convolutional Wasserstein Barycenters in POT\n' + str(nb_images) +
                ' interpolation images - regularization = ' + str(round(reg, 10)),
                fontsize=14)

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

    plt.figure(1, figsize=(5,5))
    plt.scatter(x,y,color="blue",alpha=.7, label="f(x)=exp(-x^2)")
    plt.xlabel("x in 0 to 3 step 0.03")
    plt.ylabel("Regularization input")
    plt.legend()
    plt.title("Input for barycenter's regularization parameter.")
    # plt.show()

    # Plot the distributions alone (no barycenter)
    plt.figure(2, figsize=(5, 5))
    plt.xlim(10, 90)
    plt.ylim(10, 90)
    for i in range(n_dist):
        plt.scatter(dists[i, :, 0], dists[i, :, 1], alpha=.5)
    plt.title(str(n_dist)+" bivariate normal distributions.")
    # plt.show()

    for reg in y:
        k += 1
        print("graph", k, "over", stop // step)
        # Compute the regularized (entropy) barycenter
        d = ot.bregman.convolutional_barycenter2d(dists, reg=reg)

        # Plotting
        ## Plot the distributions
        plt.figure(3, figsize=(5, 5))
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


def generate_normal(n_dist, n_pt,rng):
    dists = []
    for i in range(n_dist):
        mu = (rng.random(size=(1, 2)) - 0.5) * 40 + 50
        sigma_xy = rng.random(size=(2)) * 5 + 4
        rho = rng.randint(0, 1)
        sigma = np.array([[sigma_xy[0] ** 2, np.prod(sigma_xy) * rho],
                          [np.prod(sigma_xy) * rho, sigma_xy[1] ** 2]])
        dists.append(ggauss(n_pt, m=mu, sigma=sigma, random_state=42))
    dists = np.array(dists)
    return dists


def generate_unif(n_dist,n_pt,rng):
    dists = []
    for i in range(n_dist):
        minimum_x, minimum_y = rng.randint(0,49,size=(2))
        maximum_x, maximum_y = rng.randint(51,100,size=(2))
        X = rng.uniform(minimum_x, maximum_x, size=(n_pt))
        Y = rng.uniform(minimum_y, maximum_y, size=(n_pt))
        points = np.array([[i,j] for i,j in zip(X,Y)])
        dists.append(points)
    dists = np.array(dists)
    return dists


# @timer
def barycenter4(dist_type = "mix", n_dist = 10, n_pt = 10, seed = 42069, plot=True):
    # Initial parameters
    rng = np.random.RandomState(seed)

    # Generate the distributions
    
    if dist_type == "normal" :
        dists = generate_normal(n_dist, n_pt, rng)
    if dist_type == "uniform" :
        dists = generate_unif(n_dist,n_pt,rng)
    if dist_type == "mix" :
        dists_norm = generate_normal(n_dist//2,n_pt,rng)
        dists_unif = generate_unif(n_dist//2,n_pt,rng)
        dists = np.concatenate((dists_norm, dists_unif), axis = 0)
        rng.shuffle(dists)


    baryc = ot.bregman.convolutional_barycenter2d(dists, reg=0.0004)
    # Plot the distributions and the barycenter
    plt.figure(1, figsize=(7, 7))
    # plt.axis('off')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    for i in range(n_dist):
        plt.scatter(dists[i, :, 0], dists[i, :, 1], alpha=.25)#, label="mu_{}".format(i))
    plt.scatter(baryc[:,0], baryc[:,1], color="black", label="barycenter")
    plt.title(str(n_dist) + f" bivariate {dist_type} distributions \n Barycenter - reg = 4e-4")
    plt.legend()
    # plt.show()

    # inductive barycenter

    two_distributions = dists[:2]
    weights = np.array([1/2,1/2])
    barycenters = []
    for i in range(n_dist-1) :
        if i > 0 :
            two_distributions = np.array([b , dists[i+1]])
            weights = np.array([ 1-1/(i+2) , 1/(i+2) ])
        b = ot.bregman.convolutional_barycenter2d(two_distributions, weights=weights, reg=0.0004)
        barycenters.append(b)
    barycenters = np.array(barycenters)


    # plot steps of inductive barycenter
    if plot :
        n_row_plot = math.floor(np.sqrt(n_dist-1))
        n_col_plot = math.ceil(np.sqrt(n_dist-1))

        plt.figure(2,figsize=(5*n_col_plot,5*n_row_plot))
        for i in range(len(barycenters)) :
            plt.subplot(n_row_plot,n_col_plot,i+1)
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.axis("off")
            if i == 0 :
                plt.scatter(dists[i, :, 0], dists[i, :, 1], alpha=.25, label="mu_0", color="blue")
                plt.scatter(dists[i+1, :, 0], dists[i+1, :, 1], alpha=.25, label="mu_1", color="red")
                plt.scatter(barycenters[i,:,0], barycenters[i,:,1], color="black", label="barycenter_0")
                plt.title("Initialization - barycenter 0")
            else :
                plt.scatter(barycenters[i-1, :, 0], barycenters[i-1, :, 1], color="blue", alpha=.3,
                            label="barycenter {}".format(i - 1))
                plt.scatter(dists[i+1, :, 0], dists[i+1, :, 1], alpha=.25, label="mu_{}".format(i+1), color="red")
                plt.scatter(barycenters[i, :, 0], barycenters[i, :, 1], color="black", label="barycenter {}".format(i))
                plt.title("Induction - barycenter {}".format(i))
            plt.legend()
        plt.suptitle(f"Steps of the Barycenter by induction. {dist_type} distributions.")
        # plt.show()

        # comparison between the two methods.
        plt.figure(3,figsize=(15,7))

        plt.subplot(1,2,1)
        plt.scatter(baryc[:,0], baryc[:,1], color="black", label="Barycenter")
        for i in range(n_dist):
            plt.scatter(dists[i, :, 0], dists[i, :, 1], alpha=.2, label="mu_{}".format(i))
        plt.title("Barycenter computed directly with all distributions.")

        plt.subplot(1,2,2)
        plt.scatter(barycenters[-1,:,0], barycenters[-1,:,1], color="black", label="barycenter")
        for i in range(n_dist):
            plt.scatter(dists[i, :, 0], dists[i, :, 1], alpha=.2, label="mu_{}".format(i))
        plt.title("Barycenter computed by induction.")
        plt.suptitle(f"Comparison between the two methods. {dist_type} distributions.")
        plt.show()

    difference = abs(baryc - barycenters[-1])
    mean_diff = np.sum(difference)/n_pt
    print(f"{dist_type} distributions.")
    # print("difference between the two methods :\n",difference)
    print("Average difference :\n",mean_diff)

    return (difference,mean_diff)


@timer
def test_baryc4(n_tests = 100, seed = 4269) :
    rng = np.random.RandomState(seed)
    seeds = rng.randint(99,9999,size=(n_tests))
    print("used seeds for the tests :\n",list(seeds))

    #test with normal distributions
    t = time()
    print("-"*50,"\n\nComparing methods for normal distributions.\n")
    diff_norm = 0
    i=0
    for s in seeds :
        i+=1
        if i % 10 == 0 :
            print(i,"..",sep="", end="")
        d = barycenter4(dist_type = "normal", n_dist = 10, n_pt = 10, seed = s, plot=False)[1]
        diff_norm += d/n_tests

    print("\n\t average difference for normal distributions, over",i,"tests :\n", round(diff_norm,7))
    print("Elapsed time :",round(time()-t,3),'s.')

    #test with uniform distributions
    t = time()
    print("-"*50,"\n\nComparing methods for uniform distributions.\n")
    diff_unif = 0
    i=0
    for s in seeds[:20] :
        i+=1
        if i % 10 == 0 :
            print(i,"..",sep="", end="")
        d = barycenter4(dist_type = "uniform", n_dist = 10, n_pt = 10, seed = s, plot=False)[1]
        diff_unif += d/n_tests

    print("\n\t average difference for uniform distributions, over",i,"tests :\n", round(diff_unif,7))
    print("Elapsed time :",round(time()-t,3),'s.')

    #test with mixed distributions
    t = time()
    print("-"*50,"\n\nComparing methods for mixed distributions.\n")
    diff_mixed = 0
    i=0
    for s in seeds[:20] :
        i+=1
        if i % 10 == 0 :
            print(i,"..",sep="", end="")
        d = barycenter4(dist_type = "mix", n_dist = 10, n_pt = 10, seed = s, plot=False)[1]
        diff_mixed += d/n_tests

    print("\n\t average difference for mixed distributions, over",i,"tests :\n", round(diff_mixed,7))
    print("Elapsed time :",round(time()-t,3),'s.')







if __name__ == "__main__":
    barycenter()
