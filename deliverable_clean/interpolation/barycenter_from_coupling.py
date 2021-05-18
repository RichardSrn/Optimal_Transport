#! /usr/bin/env python3

import numpy as np

from cost_matrix import cost_matrix


def exact_barycenter(source, target, t):
    """
    coordinates of the source and the target.
    t is the weight in the formula :
        t -> argmin_mu (1-t)W_2(mu_f,mu)^2 + t*W_2(mu_g, mu)^2
    """
    return (1 - t) * source + t * target


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
    exact_baryc = exact_barycenter(mu_s_coord, mu_t_coord, 1 / 2)

    ##############################################
    # interpolate the coordinates with the grid. #
    ##############################################
    interpolated_barycenter = np.zeros(shape=(size_x, size_y))

    k = 0
    for x, y in exact_baryc:
        # For each point, if the points is not exactly on a grid's points, we interpolate it.
        # Example, b = (.2, 0) with weight 1.
        #   then we attribute +.8 on the grid's point (0,0) and +.2 on (1,0).
        # In the end we return the grid's weights.
        floor_x = int(x)
        floor_y = int(y)

        # upper_w_x stand for "upper weight for x axis".
        upper_w_x = (x - floor_x) * barycenter_weights[k]
        lower_w_x = (1 - x + floor_x) * barycenter_weights[k]
        upper_w_y = (y - floor_y) * barycenter_weights[k]
        lower_w_y = (1 - y + floor_y) * barycenter_weights[k]

        # by default we add to the floored points the corresponding weights.
        interpolated_barycenter[floor_x, floor_y] += lower_w_x * lower_w_y

        # the below conditions are just here in case one of the points is on the edge of the
        # image, to avoid an "out of bound" error.
        if floor_x < size_x:
            interpolated_barycenter[floor_x + 1, floor_y] += upper_w_x * lower_w_y
        if floor_x < size_y:
            interpolated_barycenter[floor_x, floor_y + 1] += lower_w_x * upper_w_y
        if floor_x < size_x and floor_y < size_y:
            interpolated_barycenter[floor_x + 1, floor_y + 1] += upper_w_x * upper_w_y

        k += 1

    return interpolated_barycenter


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
    barycenter = interpolate_with_mesh_grid(barycenter_coord, barycenter_weights, size_x, size_y)

    return barycenter
