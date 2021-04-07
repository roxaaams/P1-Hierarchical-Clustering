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

# def dist_mat(data,vert_start,vert_end, horz):
#     dm = np.zeros((vert_end - vert_start,horz))

#     for i in range(len(dm)):
#         for j in range(horz):
#             dm[i][j] = euclid(data[i+vert_start], data[j])
#     return dm

def dist_mat(data,vert_start,vert_end, horz):
    dm = np.zeros((vert_end - vert_start,horz))

    for i in range(len(dm)):
        for j in range(vert_start + 1,horz):
            dm[i][j] = euclid(data[i+vert_start], data[j])
    return dm


def create_split_indices(data, size = 100):
    split = len(data)//size
    print(split)
    lengths = [i for i in range (0, len(data)+1, split)] ##!!! if split != 0, not included for performance
    lengths.append(len(data)) if lengths[-1] != len(data) else None
    
    return lengths

data = np.random.rand(3000,2)

ind_list = create_split_indices(data)

jobs = -1


start = time.time()

dm_joblib = Parallel(n_jobs = jobs)(delayed(dist_mat)(data, ind_list[i-1], ind_list[i], len(data)) for i in range(1,len(ind_list))) 


stop = time.time()


    
print("dm_joblib ", stop - start)

finalize = dm_joblib[0]
for i in range(1, len(dm_joblib)):
    finalize = np.vstack((finalize,dm_joblib[i]))


start = time.time()
distanceMatrix = pair_dist(data, n_jobs = jobs)
stop = time.time()

print("godlike: ", stop - start)

print(finalize)
print(distanceMatrix)