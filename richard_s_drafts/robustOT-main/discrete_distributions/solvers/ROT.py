# Code for robust optimal transport formulation

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import plotter
from pathlib import Path
import os


def hist_from_images(img1: np.ndarray, img2: np.ndarray):
    """
    Turns a 2D image into a 1D vector -a histogram-.
    """

    # turn the 2D images images into 1D histograms
    hist1 = img1.reshape((np.prod(img1.shape, -1)))
    hist2 = img2.reshape((np.prod(img2.shape, -1)))

    return (hist1, hist2)


class ROTSolver(object):
    def __init__(self, dist1, dist2, marginal1=None, marginal2=None, ground_cost='l2', rho=0.1, logdir='results'):
        self.dist1 = dist1
        self.dist2 = dist2
        self.rho = rho
        nsamples1 = dist1.shape[0]
        nsamples2 = dist2.shape[0]
        self.nsamples1 = nsamples1
        self.nsamples2 = nsamples2

        if marginal1 is None:
            self.marginal1 = np.array([1/nsamples1 for i in range(nsamples1)])
        else:
            self.marginal1 = marginal1

        if marginal2 is None:
            self.marginal2 = np.array([1/nsamples2 for i in range(nsamples2)])
        else:
            self.marginal2 = marginal2
        self.marginal1 = np.expand_dims(self.marginal1, axis=1)
        self.marginal2 = np.expand_dims(self.marginal2, axis=1)

        self.ground_cost = ground_cost
        assert ground_cost in ['l2']
        self.logdir = logdir
        Path(self.logdir).mkdir(parents=True, exist_ok=True)

    def form_cost_matrix(self, x, y):
        if self.ground_cost == 'l2':
            return np.sum(x ** 2, 1)[:, None] + np.sum(y ** 2, 1)[None, :] - 2 * x.dot(y.transpose())

    def solve(self, plot=True):
        C = self.form_cost_matrix(self.dist1, self.dist2)
        P = cp.Variable((self.nsamples1, self.nsamples2))
        a_tilde = cp.Variable((self.nsamples1, 1))
        b_tilde = cp.Variable((self.nsamples2, 1))

        u = np.ones((self.nsamples2, 1))
        v = np.ones((self.nsamples1, 1))
        constraints = [0 <= P, cp.matmul(P, u) == a_tilde, cp.matmul(P.T, v) == b_tilde, 0 <= a_tilde, 0 <= b_tilde]
        constraints.append(cp.sum([((self.marginal1[i] - a_tilde[i]) ** 2) / self.marginal1[i]
                                   for i in range(self.nsamples1)]) <= self.rho)
        constraints.append(cp.sum([((self.marginal2[i] - b_tilde[i]) ** 2) / self.marginal2[i]
                                   for i in range(self.nsamples1)]) <= self.rho)

        objective = cp.Minimize(cp.sum(cp.multiply(P, C)))
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver='SCS')
        coupling = P.value

        print("Number of non-zero values in P: {} (n + m-1 = %d)".format(len(coupling[coupling > 1e-5]),
                                                                         self.nsamples1 + self.nsamples2 - 1))
        print("Objective function: {}".format(objective.value))

        if plot:
            print('Generating plots ...')
            plotter.generate_scatter_plots(self.dist1, self.dist2,
                                           '{}/orig.png'.format(self.logdir))
            plotter.generate_scatter_plots_with_coupling(self.dist1, self.dist2, coupling,
                                                         '{}/coupling.png'.format(self.logdir))

        robust_OT_cost = objective.value
        return robust_OT_cost

def test_robust_ot():


    rng = np.random.RandomState(42)

    # choose 2 images
    index = rng.choice(np.arange(199), 2, replace=False)
    img1, img2 = np.load("../../../PRNI2018_TLp_bary/artificial_data_nn.npy")[index]
    size_x, size_y = img1.shape

    # make it absolute value
    img1 = abs(img1)
    img2 = abs(img2)

    # normalize the data
    img1 = img1 / img1.sum()
    img2 = img2 / img2.sum()

    # turn 2D images into 1D vector --histogram--
    # hist1, hist2 = hist_from_images(img1, img2)

    rot = ROTSolver(img1, img2)
    cost = rot.solve(plot=False)




    # get the coupling matrix
    # coupling = coupling_from_2_hist(hist1, hist2, TEST_ALG, size_x, size_y)

    # # get the barycenter
    # barycenter = barycenter_from_coupling(coupling, size_x, size_y)

test_robust_ot()  