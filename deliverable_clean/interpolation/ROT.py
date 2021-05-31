# Code for robust optimal transport formulation

from pathlib import Path

import cvxpy as cp
import numpy as np


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
        self.dist1 = dist1[None].T
        self.dist2 = dist2[None].T
        self.rho = rho
        nsamples1 = dist1.shape[0]
        nsamples2 = dist2.shape[0]
        self.nsamples1 = nsamples1
        self.nsamples2 = nsamples2

        if marginal1 is None:
            self.marginal1 = np.array([1 / nsamples1 for i in range(nsamples1)])
        else:
            self.marginal1 = marginal1

        if marginal2 is None:
            self.marginal2 = np.array([1 / nsamples2 for i in range(nsamples2)])
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

        print("\tGenerate optimization constraints.")
        constraints = [0 <= P, cp.matmul(P, u) == a_tilde, cp.matmul(P.T, v) == b_tilde, 0 <= a_tilde, 0 <= b_tilde]
        constraints.append(cp.sum([((self.marginal1[i] - a_tilde[i]) ** 2) / self.marginal1[i]
                                   for i in range(self.nsamples1)]) <= self.rho)
        constraints.append(cp.sum([((self.marginal2[i] - b_tilde[i]) ** 2) / self.marginal2[i]
                                   for i in range(self.nsamples1)]) <= self.rho)
        print("\tDONE.")

        print("\tGenerate optimization problem.")
        objective = cp.Minimize(cp.sum(cp.multiply(P, C)))
        prob = cp.Problem(objective, constraints)
        print("\tDONE.")

        print("\tSolve optimization problem.")
        result = prob.solve(solver='SCS')
        print("\tDONE.")

        coupling = P.value

        robust_OT_cost = objective.value
        return (robust_OT_cost, coupling)


def ROT(hist1: np.ndarray, hist2: np.ndarray, *args):
    """
    Adapted version of the robustOT algorithm to take as input 1D distributions,
    instead of d-dimensional distributions, d>=2.

    Inputs :
        hist1, hist2 : np.ndarray, size (n,1)
    output :
        coupling : np.ndarray, size (n,n), coupling matrix for the n points of the histograms.
    """

    rot = ROTSolver(hist1, hist2)
    _, coupling = rot.solve(plot=False)

    return coupling
