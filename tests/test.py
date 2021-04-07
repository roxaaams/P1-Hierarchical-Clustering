# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:20:30 2021

@author: pault
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

### Using a "linkagematrix" to later plot dendrogram from scipy.cluster.hierarchy and keep track of clusters
### as described here: https://stackoverflow.com/questions/9838861/scipy-linkage-format
### and here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html


def euclidian_dist(a,b):
    return np.sum(np.square(a-b)) ** (1/2)


## TODO combine with creation of distance matrix (build and return first)

def completeLinkage(distMat): 
    all_row_pos = tuple()
    all_row_val = 10^5
    for i in range(len(distMat)):
        one_row_val = 0
        one_row_pos = tuple()
        for j in range(i+1,len(distMat)):   ##There will be an error because out of range (just treat edge-cases)
            elem = distMat[i][j]
            if(elem >= one_row_val):
                one_row_val = elem
                one_row_pos = (i,j)
        if(one_row_val <= all_row_val):
            all_row_val = one_row_val
            all_row_pos = one_row_pos
    
    return [all_row_pos, all_row_val]

## TODO another method for picking element in updated dist_mat


def update_distance_matrix(dist_mat, pos): #val?
    new_cluster= []
    for x,y in zip(dist_mat[pos[0]], dist_mat[pos[1]]):
        new_cluster.append(x if x > y else y)
     
    dist_mat = np.vstack((dist_mat,new_cluster))
    new_cluster.append(0)
    new_cluster = np.array(new_cluster).reshape(len(new_cluster),1)
    
    dist_mat = np.hstack((dist_mat,new_cluster))
    ##TODO
    #thar be bugs
    dist_mat[pos[0]] = [np.inf for i in dist_mat[pos[0]]]
    dist_mat[pos[1]] = [np.inf for i in dist_mat[pos[1]]]    
    dist_mat[:,pos[0]] = [np.inf for i in dist_mat[:,pos[0]]]
    dist_mat[:,pos[1]] = [np.inf for i in dist_mat[:,pos[1]]]

    print(new_cluster)
    return dist_mat



def hierarchicalClustering(data):
    
    #will be returned, used for the dendrogram
    linkageMatrix = []
    
    number_elements_in_cluster = list(np.ones(len(data)))
    
    distanceMatrix = np.empty((len(data),len(data)))
    
    #fill distance matrix
    
    for i in range(len(data)):
        for j in range(len(data)):
            distanceMatrix[i][j] = euclidian_dist(data[i], data[j])
    print(distanceMatrix)
    
    #pick element
    [pos,val] = completeLinkage(distanceMatrix)
    
    
    
    
    elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
    linkageMatrix = np.vstack(linkageMatrix, [pos[0] + 1, pos[1] + 1, val, elements_in_new_cluster]) # pos[0] + 1 and pos[1] + 1?
    number_elements_in_cluster.append(elements_in_new_cluster)
    
    
    
    
    
    
    
    
    return

