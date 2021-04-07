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

def find_min_parallel(dm, offset):
    pos_np = np.where(dm == np.amin(dm[dm.astype(bool)]))
    if(len(pos_np[0]) > 1):
        print("here")
        pos = pos_np[0]
        val_np = dm[pos[0], pos[1]]
        pos[0] += offset
    else:
        print("there")
        val_np = dm[pos_np]
        pos_np = [pos_np[0], pos_np[1]]
        pos_np[0] += offset
    return [pos_np, val_np]


def create_split_indices(data, size = 100):
    split = len(data)//size
    #split = 1 if split == 0 else None #not included for performance
    print(split)
    lengths = [i for i in range (0, len(data)+1, split)] 
    lengths.append(len(data)) if lengths[-1] != len(data) else None
    
    return lengths


data = np.random.rand(10,2)

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

# start_c = time.time()
# [pos_c, val_c] = completeLinkage(dm)
# stop_c = time.time()

# print("classic: ", stop_c - start_c, "val: ", val_c, "pos: ", pos_c)

# start_np = time.time()
# pos_np = np.where(dm == np.amin(dm[dm.astype(bool)]))
# print(dm[dm.astype(bool)])
# stop_np = time.time()
# print(pos_np)
# pos_np = pos_np[0]
# val_np = dm[pos_np[0], pos_np[1]]
# print("np: ", stop_np - start_np, "val: ", val_np, "pos: ", pos_np) # 



start = time.time()

poss = Parallel(n_jobs = jobs, prefer = "threads") (delayed(find_min_parallel) (dm[ind[i-1]:ind[i]], i-1) for i in range(1,len(ind)))
for i in poss:
    print(poss)

stop = time.time()
print(stop - start)

# split = 
# start = time.time()
# poss = Parallel(n_jobs = jobs, prefer = "threads") (delayed(fmp) (dm[i], i) for i in range(len(dm)))
# stop = time.time()
# print(stop - start)




# ind = create_split_indices(dm)
# test = dm[ind[0]:ind[1]]
# where = np.where(test == np.amin(test[test.astype(bool)])) 
# amin = np.amin(test[test.astype(bool)])
# for i in where:
#     print(i)
# print(test)
# print(where)
# print(amin)
# print(test[where])
# print(dm[where])
# print(test[where[0], where[1]])

# poss = []
# for i in range(1, len(data)):
#     test = dm[ind[i-1]:ind[i]]
#     pos_np = np.where(test == np.amin(test[test.astype(bool)]))
#     print(pos_np)
#     print(test)
#     poss.append(find_min_parallel(test, i))
    