# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:26:23 2021

@author: pault
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances as pair_dist

import time




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

# def find_min_parallel(dm, offset):
#     pos_np = np.where(dm == np.amin(dm[dm.astype(bool)]))
#     if(len(pos_np[0]) > 1):
#         print(pos_np)
#         pos = pos_np[0]
        
#         val = dm[pos[0], pos[1]]
#         pos[0] += offset
#     else:
#         val = dm[pos_np]
#         pos = [pos_np[0], pos_np[1]]
#         pos[0] += offset
#     return [pos, val]
def find_min_parallel(dm, offset):
    pos_np = np.where(dm == np.amin(dm[dm.astype(bool)]))
    val = dm[pos_np]
    #pos = pos_np[0].tolist() + pos_np[1].tolist()
    pos = [pos_np[0][0], pos_np[1][0]]
    pos[0] += offset
    return [pos, val[0]]


def create_split_indices(data, amount = 100):
    size = len(data)//amount
    #size = 1 if size == 0 else None #not included for performance
    print(size)
    indices = [i for i in range (0, len(data)+1, size)] 
    indices.append(len(data)) if indices[-1] != len(data) else None
    
    return indices

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

data = np.random.rand(4000,2)

 
ind = create_split_indices(data)
jobs = -1
dm = Parallel(n_jobs = jobs)(delayed(dist_mat)(data, ind[i-1], ind[i], len(data)) for i in range(1,len(ind)))

dm = np.vstack(dm)

start1 = time.time()
poss = Parallel(n_jobs = jobs, prefer = "threads") (delayed(find_min_parallel) (dm[ind[i-1]:ind[i]], ind[i-1]) for i in range(1,len(ind)))
pos = []
val = 10000
for i in poss:
    print(i)
    if (i[1] <= val):
        pos = i[0]
        val = i[1]


stop1 = time.time()

# for i in range(1, len(ind)):
#     test = dm[ind[i-1]:ind[i]]
#     #print(test[:2])
#     pos_np = np.where(test == np.amin(test[test.astype(bool)]))
#     print("eyo", pos_np)
#     #print(test)
start2 = time.time()    
print("Check: ", np.amin(dm[dm.astype(bool)]) , np.where(dm == np.amin(dm[dm.astype(bool)])))#
stop2 = time.time()
#print(dm)

print(pos, val)

print(stop1 - start1)
print(stop2 - start2)
