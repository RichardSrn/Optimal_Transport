#! /usr/bin/env python3

import numpy as np

from cost_matrix import cost_matrix


def exact_barycenter(source, target, t):
    """
    coordinates of the source and the target.
    t is the weight in the formula :
        t -> argmin_mu (1-t)W_2(mu_f,mu)^2 + t*W_2(mu_g, mu)^2
    """
    return t * source + (1 - t) * target


def interpolate_with_mesh_grid(barycenter_coord, barycenter_weights, size_x, size_y):
    """
    Align the interpolation on the grid.
    As some values of the exact_barycenter may be decimal, we need to interpolate them with the grid.
    for example :
        If a value has weight 1 on coordinate (0.5,0), this algorithm will output a weight of .5 on the coordinates (0,0) and (1,0).
    """

    # get the grid
    _, grid = cost_matrix(size_x, size_y)

    # get the index of the barycenter's points in the grid array.
    mu_s_index, mu_t_index = barycenter_coord[0], barycenter_coord[1]

    # get the coordinates of the barycenter's points in the mesh.
    mu_s_coord, mu_t_coord = grid[mu_s_index], grid[mu_t_index]

    # compute the exact barycenter's coordinates in the 2D space.
    print("\tCompute exact, euclidean, barycenter.")
    exact_baryc = exact_barycenter(mu_s_coord, mu_t_coord, 1 / 2)
    print("\tDONE.")

    ##############################################
    # interpolate the coordinates with the grid. #
    ##############################################
    print("\tInterpolate the exact barycenter with the grid.")
    interpolated_barycenter = np.zeros(shape=(size_x, size_y))

    k = 0
    l = len(exact_baryc)
    progress = .2 #loop progress in percent
    for x, y in exact_baryc:
        # For each point, if the points is not exactly on a grid's points, we interpolate it.
        # Example, b = (.2, 0) with weight 1.
        #   then we attribute +.8 on the grid's point (0,0) and +.2 on (1,0).
        # In the end we return the grid's weights.

        # f_x stands for "floor coordinate on x axis"
        f_x = int(x)
        f_y = int(y)

        # u_x stands for "upper weight on x axis", etc.
        l_x = f_x+1 - x #f_x+1 = ceiling(x)
        u_x = x     - f_x
        l_y = f_y+1 - y
        u_y = y     - f_y

        # by default we add to the floored points the corresponding weights.
        interpolated_barycenter[f_x,f_y] += l_x * l_y * barycenter_weights[k]

        # the below conditions are just here in case one of the points is on the edge of the
        # image, to avoid an "out of bound" error.

        if f_x < size_x-1: # if not on right edge of the image
            interpolated_barycenter[f_x + 1, f_y] += u_x * l_y * barycenter_weights[k]

        if f_y < size_y-1: # if not on top edge of the image
            interpolated_barycenter[f_x, f_y + 1] += l_x * u_y * barycenter_weights[k]

        if f_x < size_x-1 and f_y < size_y-1: # if not on top right corner of the image
            interpolated_barycenter[f_x + 1, f_y + 1] += u_x * l_y * barycenter_weights[k]

        if k/l > progress :
            progress += .2
            print("\t\t",int(k/l*100),"%...")
        k += 1
    print("\t\t 100 %")
    print("\tDONE.")

    return interpolated_barycenter

def histogram_barycenter(barycenter_coord, barycenter_weights, size_x, size_y) :
    exact_baryc = exact_barycenter(barycenter_coord[0], barycenter_coord[1], 1/2)

    baryc = np.zeros(shape = (size_x*size_y,1))

    k = 0
    for x in exact_baryc :
        floor = int(x)

        lower = floor+1 - x
        upper = x       - floor

        baryc[floor] += barycenter_weights[k]*lower
        if upper > 0 :
            baryc[floor+1] += barycenter_weights[k]*upper

        k += 1

    return baryc

def barycenter_from_coupling(coupling: np.ndarray, size_x, size_y):
    """
    Computes the barycenter of the two histograms given the coupling matrix.
    size_x, size_y are original images' shape.
    """
    # get all the weights of the barycenter's points.
    barycenter_weights = coupling[coupling != 0]

    # get the coordinates/index of the barycenter's points in the coupling matrix.
    barycenter_coord = np.where(coupling != 0)
    barycenter_coord = np.array([barycenter_coord[0], barycenter_coord[1]])

    # Turn the barycenter's coupling matrix index into actual 2D space coordinates,
    # then interpolate the exact coordinates of the points with the grid
    # and attribute the weight of each point.
    # barycenter = interpolate_with_mesh_grid(barycenter_coord, barycenter_weights, size_x, size_y)


    hist_baryc = interpolate_with_mesh_grid(barycenter_coord, barycenter_weights, size_x, size_y)
    # barycenter = hist_baryc.reshape(size_x,size_y)

    # return barycenter
    return hist_baryc