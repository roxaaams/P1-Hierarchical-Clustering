import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize, precision=2)

def update_distance_matrix(dist_mat, pos, a):
    '''
    Updates a distance matrix with merged cluster and sets the two clusters to be merged to 0.
    It is done this way to allow the simple creation of a linkage matrix.
    Parameters
    ----------
    dist_mat : 2d-array
        A upper triangular distance matrix.
    pos : [int,int]
        Contains the indices of the two clusters to be merged.
    Returns
    -------
    dist_mat : 2d-array
        The updated distance-matrix
    '''
    for i in range(len(dist_mat)):
        if(i == pos[0] or i == pos[1]):
            continue
        max_ = max(dist_mat[i][pos[0]], dist_mat[i][pos[1]])
        dist_mat[i][pos[0]] = max_
        dist_mat[i][pos[1]] = max_
        dist_mat[:, i][pos[0]] = max_
        dist_mat[:, i][pos[1]] = max_

    dist_mat[pos[1]] = [0 for i in range(len(dist_mat))]
    dist_mat[:, pos[1]] = [0 for i in range(len(dist_mat))]

    return dist_mat

def complete_linkage(cluster1, cluster2, distance_matrix):
    distances = []

    point_x = min(cluster1)
    point_y = min(cluster2)
    distances.append(
        distance_matrix[min(point_x, point_y)][max(point_x, point_y)])

    return max(distances)