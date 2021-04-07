# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:26:23 2021

@author: pault
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances as pair_dist

import time



def cL_parallel3_base(elem, offset): 
    pos = tuple()
    val = np.inf
    for i in range(offset, len(elem)): #-offset
        if(elem[i] < val and elem[i] > 0):
            val = elem[i]
            pos = (offset, i)   #switched to have equal outcomes to non-parallel
    return [pos, val]

# def find_min_parallel(dm, offset):
#     pos_np = np.where(dm == np.amin(dm[dm.astype(bool)]))
#     if(len(pos_np[0]) > 1):
#         pos = pos_np[0]
#         val_np = dm[pos[0], pos[1]]
#         pos[0] += offset
#     else:
#         val_np = dm[pos_np]
#         pos_np = [pos_np[0], pos_np[1]]
#         pos_np[0] += offset
#     return [pos_np, val_np]

def find_min_parallel(dm, offset):
    pos_np = np.where(dm == np.amin(dm[dm.astype(bool)]))
    if(len(pos_np[0]) > 1):
        pos = pos_np[0]
        val = dm[pos[0], pos[1]]
        pos[0] += offset
    else:
        val = dm[pos_np]
        pos = [pos_np[0], pos_np[1]]
        pos[0] += offset
    return [pos, val]


def create_split_indices(data, amount = 100):
    size = len(data)//amount
    #size = 1 if size == 0 else None #not included for performance
    print(size)
    indices = [i for i in range (0, len(data)+1, size)] 
    indices.append(len(data)) if indices[-1] != len(data) else None
    
    return indices



data = np.random.rand(100,2)

dm = pair_dist(data)
ind = create_split_indices(dm,2)
jobs = -1

start_p = time.time()
poss = Parallel(n_jobs = jobs) (delayed(cL_parallel3_base) (dm[i], i) for i in range(len(data)))


pos = tuple()
val = np.inf

for i in poss:
    if(i[1] <= val): 
        pos = i[0]
        val = i[1]

stop_p = time.time()

print("parallel: ", stop_p - start_p, "val: ", val, "pos: ", pos)

start = time.time()

poss = Parallel(n_jobs = jobs, prefer = "threads") (delayed(find_min_parallel) (dm[ind[i-1]:ind[i]], ind[i-1]) for i in range(1,len(ind)))


for i in range(1, len(ind)):
    test = dm[ind[i-1]:ind[i]]
    #print(test[:2])
    pos_np = np.where(test == np.amin(test[test.astype(bool)]))
    print(pos_np)
    #print(test)
    
print("Check: ", np.amin(dm[dm.astype(bool)]) , np.where(dm == np.amin(dm[dm.astype(bool)])))#
#print(dm)

for i in poss:
    print(i)
print(ind)