# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 23:01:04 2021

@author: pault
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

from sklearn.metrics import pairwise_distances as pair_dist

import time


from joblib import Parallel, delayed

def euclidian_dist(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** (1/2)

def find_min_parallel(distance_matrix, offset):
    if(not np.count_nonzero(distance_matrix)): return [[-1,-1], np.inf]
    pos_np = np.where(distance_matrix == np.amin(distance_matrix[distance_matrix.astype(bool)]))
    val = distance_matrix[pos_np]
    pos = [pos_np[0][0], pos_np[1][0]]
    pos[0] += offset
    return [pos, val[0]]

def get_next(distance_matrix, jobs):
    start = time.time()
    ind = create_split_indices(distance_matrix)

    poss = Parallel(n_jobs = jobs, prefer = "threads") (delayed(find_min_parallel) (distance_matrix[ind[i-1]:ind[i]], ind[i-1]) for i in range(1,len(ind)))
    pos = []
    val = 10000
    for i in poss:

        if (i[1] <= val):
            pos = i[0]
            val = i[1]
    end = time.time()
    #print("get_next: ", end -start)
    return [pos, val]

def dist_mat(data,vert_start,vert_end, horz):
    distance_matrix = np.zeros((vert_end - vert_start,horz))
    for i in range(len(distance_matrix)):
        for j in range(vert_start + i,horz):
            distance_matrix[i][j] = euclidian_dist(data[i+vert_start], data[j])
    return distance_matrix

def create_split_indices(data, amount = 100):
    size = len(data)//amount
    size = 1 if size == 0  else size #not included for performance

    indices = [i for i in range (0, len(data)+1, size)] 
    indices.append(len(data)) if indices[-1] != len(data) else None
    
    return indices

def update_distance_matrix(dist_mat, pos):
    start = time.time()
    to_merge_1 = [x if x > y else y for x,y in zip(dist_mat[pos[0]],dist_mat[:,pos[0]])]
    to_merge_2 = [x if x > y else y for x,y in zip(dist_mat[pos[1]],dist_mat[:,pos[1]])]
    new_cluster = [x if x > y else y for x,y in zip(to_merge_1,to_merge_2)]
    new_cluster.append(0)
    new_cluster = np.array(new_cluster).reshape(len(new_cluster),1)
    dist_mat = np.vstack((dist_mat, np.zeros(len(dist_mat))))
    dist_mat = np.hstack((dist_mat, new_cluster))
    dist_mat[pos[0]] = [0 for i in dist_mat[pos[0]]]
    dist_mat[pos[1]] = [0 for i in dist_mat[pos[1]]]    
    dist_mat[:,pos[0]] = [0 for i in dist_mat[:,pos[0]]]
    dist_mat[:,pos[1]] = [0 for i in dist_mat[:,pos[1]]]
    stop = time.time()
    #print("update: ", stop-start)
    return dist_mat


def hierarchicalClustering(data):
  
    #where to split the data for parallelization
    ind = create_split_indices(data)

    #set number of processors
    jobs = -1

    #fill distance matrix
    start = time.time()
    distance_matrix = Parallel(n_jobs = jobs)(delayed(dist_mat)(data, ind[i-1], ind[i], len(data)) for i in range(1,len(ind)))
    stop = time.time()
    print("dm: ", stop-start)
    #put together elements of distance_matrix
    distance_matrix = np.vstack(distance_matrix)

    
    

    #pick element
    [pos,val] = get_next(distance_matrix, jobs)

    
    
    number_elements_in_cluster = list(np.ones(len(data)))
    elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
    linkage_matrix =  [pos[0] , pos[1] , val, elements_in_new_cluster]
    number_elements_in_cluster.append(elements_in_new_cluster)
    

    distance_matrix = update_distance_matrix(distance_matrix, pos)


    while(number_elements_in_cluster[-1] < len(data)):
        start = time.time()
        [pos,val] = get_next(distance_matrix, jobs)
        elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
        linkage_matrix = np.vstack((linkage_matrix, [pos[0] , pos[1] , val, elements_in_new_cluster])) 
        number_elements_in_cluster.append(elements_in_new_cluster)
        distance_matrix = update_distance_matrix(distance_matrix, pos)
        stop = time.time()
        print("iter : ", stop - start)
    return linkage_matrix


test_data = np.random.rand(3000,2)

lm = hierarchicalClustering(test_data)


plt.figure(figsize =(15, 15)) 
plt.title('my') 
Dendrogram = shc.dendrogram(lm)
# plt.title('test') 
# Dendrogram = shc.dendrogram((shc.linkage(test_data, method ='complete', metric = "euclidean")))

print(lm)
#print(lm ==shc.linkage(test_data, method ='complete', metric = "euclidean"))
print(all([all(x == y) for x,y in zip(lm,shc.linkage(test_data, method ='complete', metric = "euclidean"))]))