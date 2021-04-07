# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 02:25:43 2021

@author: pault
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances as pair_dist

import time

def dist_mat(data,vert_start,vert_end, horz):
    dm = np.zeros((vert_end - vert_start,horz))

    for i in range(len(dm)):
        for j in range(vert_start + i,horz):
            dm[i][j] = euclid(data[i+vert_start], data[j])
    return dm

def euclid(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** (1/2)

def create_split_indices(data, amount = 100):
    size = len(data)//amount
    #size = 1 if size == 0 else None #not included for performance
    print(size)
    indices = [i for i in range (0, len(data)+1, size)] 
    indices.append(len(data)) if indices[-1] != len(data) else None
    
    return indices

data = np.random.rand(4,2)

 
ind = create_split_indices(data,2)
jobs = -1
dm = Parallel(n_jobs = jobs)(delayed(dist_mat)(data, ind[i-1], ind[i], len(data)) for i in range(1,len(ind)))
test = np.vstack(dm)
print(test)