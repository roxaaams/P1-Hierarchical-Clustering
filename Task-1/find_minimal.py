# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:05:07 2021

@author: pault
"""

###Functions to find the minimal element of a distance matrix

import numpy as np
from joblib import Parallel, delayed

def find_min_parallel(distance_matrix, offset):
    """
    Function to find the minimum and its position, given a part of a matrix and its offset.

    Parameters
    ----------
    distance_matrix : 2d-array 
        Part of a distance_matrix
    offset : int
        How far (vertically) the part of the matrix received is from the first row of the original matrix.

    Returns
    -------
    list
        pos = position of the minimal value
        val = minimal element in matrix-chunk

    """

    if(not np.count_nonzero(distance_matrix)): return [[-1,-1], np.inf]

    val = np.amin(distance_matrix[distance_matrix.astype(bool)])

    pos = np.where(distance_matrix == val)

    pos = [pos[0][0], pos[1][0]]
    pos[0] += offset
    
    return [pos, val]

    

def find_min_wrapper(distance_matrix, jobs):
    """
    "Wrapper" function to find minimum element and its position in a full distance-matrix.

    Parameters
    ----------
    distance_matrix : 2d-array
        A full distance-matrix
    jobs : int
        Number of processes to be used. -1 equals all available.

    Returns
    -------
    list
        pos = position of minimal element of distance matrix.
        val = minimal element of distance matrix

    """
    ind = create_split_indices(distance_matrix)

    possibles = Parallel(n_jobs = jobs, prefer = "threads") (delayed(find_min_parallel) (distance_matrix[ind[i-1]:ind[i]], ind[i-1]) for i in range(1,len(ind)))
    pos = []
    val = np.inf
    for i in possibles:
        if (i[1] <= val):
            pos = i[0]
            val = i[1]

    return [pos, val]

def find_min_single(distance_matrix, jobs):
    """
    Function to find minimum element of distance matrix and its position, not parallelized.
    Will be used on smaller datasets to reduce overhead of parallelization

    Parameters
    ----------
    distance_matrix : 2d array
        A full distance matrix.
    jobs : int
        Not used in function, included to simply switch between functions.

    Returns
    -------
    list
        val = minimal element of distance_matrix.
        pos = position of val in matrix.

    """
    val = np.amin(distance_matrix[distance_matrix.astype(bool)])

    pos = np.where(distance_matrix == val)

    pos = [pos[0][0], pos[1][0]]
    
    return [pos, val]

def create_split_indices(data, amount = 100):
    """
    Simple function to create indices to split data for parallelization. 

    Parameters
    ----------
    data : 2d-array
        Data to be split.
    amount : int, optional
        Amount of "chunks" to be created. The default is 100.

    Returns
    -------
    indices : list
        List of indices on where to split data.

    """
    size = len(data)//amount
    size = 1 if size == 0  else size 

    indices = [i for i in range (0, len(data), size)] 
    if(indices[-1] != len(data)): indices.append(len(data))
    
    return indices
