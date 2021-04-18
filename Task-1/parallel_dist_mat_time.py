# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 00:07:34 2021

@author: pault
"""

from joblib import Parallel, delayed
import numpy as np
import time
from sklearn.metrics import pairwise_distances as pair_dist

def euclid(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** (1/2)

def dist_mat_parallel(data,vert_start,vert_end, horz):
    dm = np.zeros((vert_end - vert_start,horz))

    for i in range(len(dm)):
        for j in range(vert_start + 1,horz):
            dm[i][j] = euclid(data[i+vert_start], data[j])
    return dm

def dist_mat_singular(data):
    dm = np.zeros((len(data),len(data)))
    for i in range(len(dm)):
        for j in range (i + 1,len(dm)):
            dm[i][j] = euclid(data[i], data[j])

def create_split_indices(data, size = 100):
    split = len(data)//size
    if(split == 0): split = 1
    lengths = [i for i in range (0, len(data)+1, split)] ##!!! if split != 0, not included for performance
    lengths.append(len(data)) if lengths[-1] != len(data) else None
    
    return lengths


jobs = -1

for i in range(100,1501,100):
    data = np.random.rand(i,2)
    start_singular = time.time()
    dm_singular = dist_mat_singular(data)
    stop_singular = time.time()
    ind_list = create_split_indices(data, i//10)
    start_parallel = time.time()
    dm_parallel =  Parallel(n_jobs = jobs)(delayed(dist_mat_parallel)(data, ind_list[i-1], ind_list[i], len(data)) for i in range(1,len(ind_list)))
    stop_parallel = time.time()
    start_prof = time.time()
    dm_prof = pair_dist(data)
    stop_prof = time.time()
    print("size: ", i)
    print("singular: ", stop_singular - start_singular)
    print("parallel: ", stop_parallel - start_parallel)
    print("professional: ", stop_prof - start_prof)

    

    

