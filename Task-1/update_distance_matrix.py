# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:09:12 2021

@author: pault
"""

def update_distance_matrix(dist_mat, pos):
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
        if(i == pos[0] or i == pos[1]): continue
        max_ = max(dist_mat[i][pos[0]], dist_mat[i][pos[1]])
        dist_mat[i][pos[0]] = max_
        dist_mat[i][pos[1]] = max_
        dist_mat[:,i][pos[0]] = max_
        dist_mat[:,i][pos[1]] = max_

    dist_mat[pos[1]] = [0 for i in range(len(dist_mat))]
    dist_mat[:,pos[1]] = [0 for i in range(len(dist_mat))]

    return dist_mat
