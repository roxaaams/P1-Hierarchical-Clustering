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

### Using a "linkagematrix" to later plot dendrogram from scipy.cluster.hierarchy and keep track of clusters
### as described here:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
### and here: https://stackoverflow.com/questions/9838861/scipy-linkage-format


from joblib import Parallel, delayed

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

def find_min_parallel(distance_matrix, offset):
    '''
    

    Parameters
    ----------
    distance_matrix : Part of the distance Matrix.
    offset : Since this function only receives only a part of the distance Matrix, the
                offset is needed to get the correct position.

    Returns
    -------
    list
        pos = position of the minimal Value
        val = minimal value

    '''
    if(not np.count_nonzero(distance_matrix)): return [[-1,-1], np.inf]
    pos_np = np.where(distance_matrix == np.amin(distance_matrix[distance_matrix.astype(bool)]))
    val = distance_matrix[pos_np]
    pos = [pos_np[0][0], pos_np[1][0]]
    pos[0] += offset
    return [pos, val[0]]

def find_min(distance_matrix, jobs):
    '''
    "Wrapper" function to find a minimum in parallel.

    Parameters
    ----------
    distance_matrix : A full, upper triangular distance matrix; 2d array
    jobs : How many processors should be used. -1 equals all.

    Returns
    -------
    list
        pos = position of the minimal Value
        val = minimal value

    '''
    
    ind = create_split_indices(distance_matrix)

    possibles = Parallel(n_jobs = jobs, prefer = "threads") (delayed(find_min_parallel) (distance_matrix[ind[i-1]:ind[i]], ind[i-1]) for i in range(1,len(ind)))
    pos = []
    val = np.inf
    for i in possibles:

        if (i[1] <= val):
            pos = i[0]
            val = i[1]

    return [pos, val]

def dist_mat_parallel(data,vert_start,vert_end, horz):
    '''
    Used for parallel computation of an upper triangular distance matrix

    Parameters
    ----------
    data : part of the data for which a distance matrix should be created
    vert_start : marks the original beginning of "data", vertically 
    vert_end : marks the original ending of "data", vertically
    horz : length of the distance_matrix to be created

    Returns
    -------
    distance_matrix : part of the distance matrix calculated from "data" received

    '''
    distance_matrix = np.zeros((vert_end - vert_start,horz))
    
    for i in range(len(distance_matrix)):
        for j in range(vert_start + i,horz):
            distance_matrix[i][j] = euclidian_dist(data[i+vert_start], data[j])
            
    return distance_matrix

def dist_mat(data, jobs):
    '''
    "Wrapper" function to create distance matrix in parallel.

    Parameters
    ----------
    data : 2d-array
        Data for which a distance matrix should be created.
    jobs : int
        Number of processors to be used. -1 equals all.

    Returns
    -------
    distance_matrix : 2d-array
        An upper triangular distance matrix.

    '''
    ind = create_split_indices(data)
    distance_matrix = Parallel(n_jobs = jobs)(delayed(dist_mat_parallel)(data, ind[i-1], ind[i], len(data)) for i in range(1,len(ind)))
    distance_matrix = np.vstack(distance_matrix)
    return distance_matrix

def create_split_indices(data, amount = 100):
    '''
    Function to create indices to use when splitting up the data for any parallel function.    
    
    Parameters
    ----------
    data : Data to be split up
    amount : int, optional
        Number of chunks to be created. The default is 100.

    Returns
    -------
    indices : list
        Indices which can be used to split data for parallelization.

    '''
    size = len(data)//amount
    size = 1 if size == 0  else size 

    indices = [i for i in range (0, len(data)+1, size)] 
    indices.append(len(data)) if indices[-1] != len(data) else None
    
    return indices

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
    
    return dist_mat


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
    #set number of processors
    jobs = -1
    
    #create distance matrix
    
    ## uncomment this to use my parallel approach. It is not extremely quick but was embarrassingly much work so I wanted to include it.
    # distance_matrix = dist_mat(data, jobs)
    ##
    
    distance_matrix = np.triu(pair_dist(data))
    

    #pick element
    [pos,val] = find_min(distance_matrix, jobs)

    
    #first iteration done manually
    number_elements_in_cluster = list(np.ones(len(data)))
    elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
    linkage_matrix =  [pos[0] , pos[1] , val, elements_in_new_cluster]
    number_elements_in_cluster.append(elements_in_new_cluster)
    

    distance_matrix = update_distance_matrix(distance_matrix, pos)

    # iterate until all elements are in one cluster.
    while(number_elements_in_cluster[-1] < len(data)):
        [pos,val] = find_min(distance_matrix, jobs)
        elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
        linkage_matrix = np.vstack((linkage_matrix, [pos[0] , pos[1] , val, elements_in_new_cluster])) 
        number_elements_in_cluster.append(elements_in_new_cluster)
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


result = hierarchical_clustering(test_data)

plt.figure(figsize =(15, 15)) 
plt.title('my') 
Dendrogram = shc.dendrogram(result)


#test_data = np.random.rand(200,15)





### FOR TESTING
# This sometimes prints false, especially when using pair_dist for the distance matrix, but this is because of rounding \
# errors and doesn't really make any difference.

# professional_implementation = shc.linkage(test_data, method ='complete', metric = "euclidean")
# tests = [(x,y) for i in range(len(professional_implementation)) for x,y in zip(result[i],professional_implementation[i]) if x != y ]
# print(tests)
# print(all([all(x == y) for x,y in zip(result,professional_implementation)]))
