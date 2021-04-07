# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 02:46:34 2021

@author: pault
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances as pair_dist

import time

def completeLinkage(dist_mat):
    pos = tuple()
    val = 10^5
    for i in range(len(dist_mat)):
        for j in range(i+1, len(dist_mat)):
            elem = dist_mat[i][j]
            if(elem < val and elem > 0):
                val = elem
                pos = (i,j)
        #print(pos, val)
    return [pos, val]

def cL_parallel(dm, i_,j_):
    pos = tuple()
    val = np.inf
    for i in range(i_,j_):
        for j in range(i+1, len(dm)):
            elem = dm[i][j]
            if(elem < val and elem > 0):
                val = elem
                pos = (i,j)
    return [pos, val]

def cL_parallel2(dm, vert):
    pos = tuple()
    val = np.inf
    for i in range(len(dm)):
        for j in range(i+1, len(dm[0])):
            elem = dm[i][j]
            if(elem < val and elem > 0):
                val = elem
                pos = (i + vert,j)
    return [pos, val]

def cL_parallel3_base(elem, offset): 
    pos = tuple()
    val = np.inf
    for i in range(offset, len(elem)): #-offset
        if(elem[i] < val and elem[i] > 0):
            val = elem[i]
            pos = (offset, i)   #switched to have equal outcomes to non-parallel
    return [pos, val]


def cL_parallel3_top(dm,vert_start, vert_end):
    pos = tuple()
    val = np.inf
    for i in range(vert_start, vert_end):
        [pos_temp, val_temp] = cL_parallel3_base(dm[i], i)
        if(val_temp < val):
            pos = pos_temp
            val = val_temp
    return [pos, val]

def create_split_indices(data, size = 100):
    split = len(data)//size
    #split = 1 if split == 0 else None #not included for performance
    print(split)
    lengths = [i for i in range (0, len(data)+1, split)] 
    lengths.append(len(data)) if lengths[-1] != len(data) else None
    
    return lengths

jobs = -1
data = np.random.rand(8000,2)
dm = pair_dist(data)
ind = create_split_indices(data)

start_p = time.time()

# poss = Parallel(n_jobs = jobs, prefer = "threads") (delayed(cL_parallel) (dm, ind[i-1], ind[i]) for i in range(1,len(ind)))
# poss = Parallel(n_jobs = jobs, prefer= "threads") (delayed(completeLinkage) (dm[ind[i-1]:ind[i]]) for i in range(1,len(ind)))
# poss = Parallel(n_jobs = jobs, prefer = "threads") (delayed(cL_parallel2) (dm[ind[i-1]:ind[i]], ind[i]) for i in range(len(ind)))
# poss = Parallel(n_jobs = jobs) (delayed(cL_parallel) (dm, i) for i in range(len(data))) ???

poss = Parallel(n_jobs = jobs) (delayed(cL_parallel3_base) (dm[i], i) for i in range(len(data))) #this the good one

# poss = Parallel(n_jobs = jobs) (delayed(cL_parallel3_top) (dm, ind[i-1], ind[i]) for i in range(1, len(ind)))




stop1_p = time.time()

pos = tuple()
val = np.inf

for i in poss:
    if(i[1] <= val): 
        pos = i[0]
        val = i[1]

stop2_p = time.time()

# start = time.time()
# [pos2, val2] = completeLinkage(dm)
# stop = time.time()


start_min = time.time()
pos_min = np.where(np.amin(dm != 0))
val_min = 0 # dm[pos_min[0]]
stop_min = time.time()






print(pos, val)
# print(pos2, val2)
print("parallel: ", "poss: ", stop1_p -start_p, "stack: ", stop2_p - stop1_p, "total: ", stop2_p - start_p)
# print("classis: ", stop - start)

print("numpy: ", pos_min, val_min, " time: ", stop_min - start_min)