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



def find_min(distance_matrix):
    val = np.amin(distance_matrix[distance_matrix.astype(bool)])
    pos = np.where(distance_matrix == val)
    pos = [pos[0][0], pos[1][0]] 
    return[pos, val]
    
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
  

    distance_matrix = np.triu(pair_dist(data))
    

    #pick element
    [pos,val] = find_min(distance_matrix)

    
    
    number_elements_in_cluster = list(np.ones(len(data)))
    elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
    linkage_matrix =  [pos[0] , pos[1] , val, elements_in_new_cluster]
    number_elements_in_cluster.append(elements_in_new_cluster)
    

    distance_matrix = update_distance_matrix(distance_matrix, pos)


    while(number_elements_in_cluster[-1] < len(data)):
        start = time.time()
        [pos,val] = find_min(distance_matrix)
        elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
        linkage_matrix = np.vstack((linkage_matrix, [pos[0] , pos[1] , val, elements_in_new_cluster])) 
        number_elements_in_cluster.append(elements_in_new_cluster)
        distance_matrix = update_distance_matrix(distance_matrix, pos)
        stop = time.time()
        print("iter : ", stop - start)
    return linkage_matrix


test_data = np.random.rand(8000,2)

lm = hierarchicalClustering(test_data)


plt.figure(figsize =(15, 15)) 
plt.title('my') 
Dendrogram = shc.dendrogram(lm)
# plt.title('test') 
# Dendrogram = shc.dendrogram((shc.linkage(test_data, method ='complete', metric = "euclidean")))

print(lm)
#print(lm ==shc.linkage(test_data, method ='complete', metric = "euclidean"))
print(all([all(x == y) for x,y in zip(lm,shc.linkage(test_data, method ='complete', metric = "euclidean"))]))