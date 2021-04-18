
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 20:34:06 2021

@author: pault
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import pandas as pd


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

from sklearn.metrics import pairwise_distances as pair_dist

from joblib import Parallel, delayed

import time

import sys

### Using a "linkagematrix" to later plot dendrogram from scipy.cluster.hierarchy and keep track of clusters
### as described here:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
### and here: https://stackoverflow.com/questions/9838861/scipy-linkage-format


def euclidian_dist(a,b):
    '''
    
    Parameters
    ----------
    a : point A, vector.
    b : point B, vector.

    Returns
    -------
    euclidian distance of A and B
    
    Note
    -------
    This is quicker than using numpy, because for this small operations the function calls to numpy create too much overhead

    '''
    
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** (1/2)


def update_distance_matrix(dist_mat, pos):
    '''
    Updates a distance matrix with merged cluster and sets the two clusters to be merged to 0.
    It is done this way to allow the simple creation of a linkage matrix.

    Parameters
    ----------
    dist_mat : 2d-array
        A upper triangular distance matrix.
    pos : [int,int]
        Contains the indices of the two clusters to be merged.

    Returns
    -------
    dist_mat : 2d-array
        The updated distance-matrix

    '''
    for i in range(len(dist_mat)):
        if(i == pos[0] or i == pos[1]): continue
        max_ = max(dist_mat[i][pos[0]], dist_mat[i][pos[1]])
        dist_mat[i][pos[0]] = max_
        dist_mat[i][pos[1]] = max_
        dist_mat[:,i][pos[0]] = max_
        dist_mat[:,i][pos[1]] = max_

    dist_mat[pos[1]] = [0 for i in range(len(dist_mat))]
    dist_mat[:,pos[1]] = [0 for i in range(len(dist_mat))]

    return dist_mat

def find_min_parallel(distance_matrix, offset):
    """
    Function to find the minimum and its position, given a part of a matrix and its offset.

    Parameters
    ----------
    distance_matrix : 2d-array 
        Part of a distance_matrix
    offset : int
        How far (vertically) the part of the matrix received is from the first row of the original matrix.

    Returns
    -------
    list
        pos = position of the minimal value
        val = minimal element in matrix-chunk

    """

    if(not np.count_nonzero(distance_matrix)): return [[-1,-1], np.inf]

    val = np.amin(distance_matrix[distance_matrix.astype(bool)])

    pos = np.where(distance_matrix == val)

    pos = [pos[0][0], pos[1][0]]
    pos[0] += offset
    
    return [pos, val]

    

def find_min_wrapper(distance_matrix, jobs):
    """
    "Wrapper" function to find minimum element and its position in a full distance-matrix.

    Parameters
    ----------
    distance_matrix : 2d-array
        A full distance-matrix
    jobs : int
        Number of processes to be used. -1 equals all available.

    Returns
    -------
    list
        pos = position of minimal element of distance matrix.
        val = minimal element of distance matrix

    """
    ind = create_split_indices(distance_matrix)

    possibles = Parallel(n_jobs = jobs, prefer = "threads") (delayed(find_min_parallel) (distance_matrix[ind[i-1]:ind[i]], ind[i-1]) for i in range(1,len(ind)))
    pos = []
    val = np.inf
    for i in possibles:
        if (i[1] <= val):
            pos = i[0]
            val = i[1]

    return [pos, val]

def find_min_single(distance_matrix, jobs):
    """
    Function to find minimum element of distance matrix and its position, not parallelized.
    Will be used on smaller datasets to reduce overhead of parallelization

    Parameters
    ----------
    distance_matrix : 2d array
        A full distance matrix.
    jobs : int
        Not used in function, included to simply switch between functions.

    Returns
    -------
    list
        val = minimal element of distance_matrix.
        pos = position of val in matrix.

    """
    val = np.amin(distance_matrix[distance_matrix.astype(bool)])

    pos = np.where(distance_matrix == val)

    pos = [pos[0][0], pos[1][0]]
    
    return [pos, val]

def create_split_indices(data, amount = 100):
    """
    Simple function to create indices to split data for parallelization. 

    Parameters
    ----------
    data : 2d-array
        Data to be split.
    amount : int, optional
        Amount of "chunks" to be created. The default is 100.

    Returns
    -------
    indices : list
        List of indices on where to split data.

    """
    size = len(data)//amount
    size = 1 if size == 0  else size 

    indices = [i for i in range (0, len(data), size)] 
    if(indices[-1] != len(data)): indices.append(len(data))
    
    return indices


def hierarchical_clustering(data):
    '''
    Perform hierarchical clustering (agglomorative) on a dataset. Returns a linkage matrix as described 
    in the beginning of this file.

    Parameters
    ----------
    data : 2d-array
        Data to be used.

    Returns
    -------
    linkage_matrix : 2d-array
        Keeps track of merges.
            l_m[0]: index of cluster 1 which is merged
            l_m[1]: index of cluster 2 which is merged
            l_m[2]: amount of elements in new cluster
            l_m[3]: value with which the clusters have been merged

    '''
    # set number of processeses, -1 equals all possible
    jobs = -1
    
    # create distance matrix
    
    distance_matrix = pair_dist(data)
    
    ##First iteration done "manually"
    
    # Choose if parallelization should be used. Parallelizing introduces overhead, so for smaller datasets it's quicker to not 
    # parallelize.
    if(len(data) <= 1500):
        find_min = find_min_single
    else: find_min = find_min_wrapper
    
    # pick element
    [pos,val] = find_min(distance_matrix, jobs)

    # array to keep track of number of elements in each cluster (in the beginning, each point is a cluster)
    number_elements_in_cluster = list(np.ones(len(data)))
    
    #amount of clusters
    total_num_clusters = len(distance_matrix)

    # array to keep track of "number" of cluster; since merging is done in-place. e.g if there are three clusters,
    # and clusters one and two get merged, the elements of cluster two are set to zero, the elements of cluster one
    # are set to the result of the (complete) linkage of the two clusters and cluster one is assigned the number four.
    # (Cluster two keeps the same "number", because it will not be picked in future iterations anyway)
    cluster_positions = [i for i in range(total_num_clusters)]
    
    # calculate the number of elements in the new cluster
    elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
    
    # necessary to create a valid linkage matrix, from a mathematical perspective there is no difference since 
    # dist_max [i][j] == dist_mat [j][i]
    # new_link is the information that is stored in the linkage-matrix
    if(cluster_positions[pos[0]] > cluster_positions[pos[1]]):
        new_link = [cluster_positions[pos[1]],cluster_positions[pos[0]], val, elements_in_new_cluster]
    else: new_link = [cluster_positions[pos[0]],cluster_positions[pos[1]], val, elements_in_new_cluster]

    linkage_matrix = new_link
    
    number_elements_in_cluster[pos[0]] = elements_in_new_cluster

    cluster_positions[pos[0]] = total_num_clusters
    distance_matrix = update_distance_matrix(distance_matrix, pos)
    
    # iterate until all elements are in one cluster.
    while(max(number_elements_in_cluster) < len(data)):
        [pos, val] = find_min(distance_matrix, jobs)

        elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
        
        if(cluster_positions[pos[0]] > cluster_positions[pos[1]]):
            new_link = [cluster_positions[pos[1]],cluster_positions[pos[0]], val, elements_in_new_cluster]
        else: new_link = [cluster_positions[pos[0]],cluster_positions[pos[1]], val, elements_in_new_cluster]
        linkage_matrix = np.vstack((linkage_matrix, new_link)) 

        number_elements_in_cluster[pos[0]] = elements_in_new_cluster
        
        total_num_clusters += 1
        cluster_positions[pos[0]] = total_num_clusters
        
        distance_matrix = update_distance_matrix(distance_matrix, pos)


    return linkage_matrix



    
raw_df = pd.read_csv('cc-data.csv')
raw_df = raw_df.drop('CUST_ID', axis = 1) 
raw_df.fillna(method ='ffill', inplace = True) 

# Standardize data
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(raw_df) 
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df) 
  
# Reducing the dimensions of the data 
pca = PCA(n_components = 2) 
test_data = pca.fit_transform(normalized_df) 

##uncomment next line to enter size of test set manually
# test_data = test_data[:800]

start = time.time()
my_result = hierarchical_clustering(test_data)
stop = time.time()
print("Our implementation: ", stop - start, "s")

plt.figure(figsize =(15, 15)) 
plt.title('Ours') 
Dendrogram = shc.dendrogram(my_result)

start = time.time()
professional_implementation = shc.linkage(test_data, method ='complete', metric = "euclidean")
stop = time.time()
print("Professional implementation: ", stop - start, "s")

##Uncomment to view dendrogramm of scipy implementation
# plt.title('Professional') 
# Dendrogram = shc.dendrogram(professional_implementation)


print(all([round(x,7) == round(y,7) 
            for i in range(len(professional_implementation)) 
            for x,y in zip(my_result[i], professional_implementation[i])]
          ))



