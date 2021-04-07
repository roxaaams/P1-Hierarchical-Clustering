# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 03:14:38 2021

@author: pault
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

#from sklearn.metrics import pairwise_distances as pair_dist

import time

from math import ceil 


from joblib import Parallel, delayed

### Using a "linkagematrix" to later plot dendrogram from scipy.cluster.hierarchy and keep track of clusters
### as described here: https://stackoverflow.com/questions/9838861/scipy-linkage-format
### and here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html



##Super slow because function calls
# def euclidian_dist(a,b):
#     return np.sqrt(np.sum(np.square(a-b)))

def euclidian_dist(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** (1/2)


def find_min_parallel(dm, offset):
    pos_np = np.where(dm == np.amin(dm[dm.astype(bool)]))
    val = dm[pos_np]
    #pos = pos_np[0].tolist() + pos_np[1].tolist()
    pos = [pos_np[0][0], pos_np[1][0]]
    print(pos)
    pos[0] += offset
    return [pos, val[0]]

def get_next(dm, jobs):
    ind = create_split_indices(dm)
    poss = Parallel(n_jobs = jobs, prefer = "threads") (delayed(find_min_parallel) (dm[ind[i-1]:ind[i]], ind[i-1]) for i in range(1,len(ind)))
    pos = []
    val = 10000
    for i in poss:
        if (i[1] <= val):
            pos = i[0]
            val = i[1]
    return [pos, val]
    
def dist_mat(data,vert_start,vert_end, horz):
    dm = np.zeros((vert_end - vert_start,horz))

    for i in range(len(dm)):
        for j in range(vert_start + i,horz):
            dm[i][j] = euclidian_dist(data[i+vert_start], data[j])
    return dm

def create_split_indices(data, amount = 100):
    size = len(data)//amount
    #size = 1 if size == 0 else None #not included for performance
    print(size)
    indices = [i for i in range (0, len(data)+1, size)] 
    indices.append(len(data)) if indices[-1] != len(data) else None
    
    return indices



# def update_distance_matrix(dist_mat, pos): #val?
    
#     start = time.time()
    
#     new_cluster= []
#     for x,y in zip(dist_mat[pos[0]], dist_mat[pos[1]]):
#         new_cluster.append(x if x > y else y)
     
#     dist_mat = np.vstack((dist_mat,new_cluster))
#     new_cluster.append(0)
#     new_cluster = np.array(new_cluster).reshape(len(new_cluster),1)
#     # print(new_cluster) 
#     dist_mat = np.hstack((dist_mat,new_cluster))
#     stop = time.time()
    
#     print("first part update: ", stop - start)
#     #thar be bugs
    
#     start = time.time()
    
#     dist_mat[pos[0]] = [0 for i in dist_mat[pos[0]]]
#     dist_mat[pos[1]] = [0 for i in dist_mat[pos[1]]]    
#     dist_mat[:,pos[0]] = [0 for i in dist_mat[:,pos[0]]]
#     dist_mat[:,pos[1]] = [0 for i in dist_mat[:,pos[1]]]

#     stop = time.time()
    
#     print("second part update: ", stop - start)
#     #print(new_cluster)
#     return dist_mat

def update_distance_matrix(dist_mat, pos):

    to_merge_1 = [x if x > y else y for x,y in zip(dist_mat[pos[0]],dist_mat[:,pos[0]])]
    to_merge_2 = [x if x > y else y for x,y in zip(dist_mat[pos[1]],dist_mat[:,pos[1]])]
    new_cluster = [x if x > y else y for x,y in zip(to_merge_1,to_merge_2)].append(0)
    dist_mat = np.hsatck((dist_mat, np.zeros(len(dist_mat))))
    dist_mat = np.vstack((dist_mat, new_cluster))
    
    dist_mat[pos[0]] = [0 for i in dist_mat[pos[0]]]
    dist_mat[pos[1]] = [0 for i in dist_mat[pos[1]]]    
    dist_mat[:,pos[0]] = [0 for i in dist_mat[:,pos[0]]]
    dist_mat[:,pos[1]] = [0 for i in dist_mat[:,pos[1]]]
    
    return dist_mat


data = np.random.rand(6,2)
ind = create_split_indices(data)
jobs = -1
dm = Parallel(n_jobs = jobs)(delayed(dist_mat)(data, ind[i-1], ind[i], len(data)) for i in range(1,len(ind)))

[pos, val] = get_next(dm,jobs)

number_elements_in_cluster = list(np.ones(len(data)))
elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
linkage_matrix =  [pos[0] , pos[1] , val, elements_in_new_cluster] 
number_elements_in_cluster.append(elements_in_new_cluster)
    
distanceMatrix = update_distance_matrix(distanceMatrix, pos)