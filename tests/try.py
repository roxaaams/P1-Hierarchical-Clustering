# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 00:29:15 2021

@author: pault
"""

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





# no ifs, zeros should work. yes,really. no if.

def euclidian_dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))


## TODO combine with creation of distance matrix (build and return first)

# def completeLinkage(distMat): 
#     all_row_pos = tuple()
#     all_row_val = 10^5
#     for i in range(len(distMat)):
#         one_row_val = -1
#         one_row_pos = tuple()
#         for j in range(i+1,len(distMat)):   ##There will be an error because out of range (just treat edge-cases) # no
#             elem = distMat[i][j]
#             if(elem >= one_row_val): #>=
#                 one_row_val = elem
#                 one_row_pos = (i,j)
#         if(one_row_val <= all_row_val and one_row_val > 0): #<=
#             all_row_val = one_row_val
#             all_row_pos = one_row_pos
    
#     return [all_row_pos, all_row_val]

def completeLinkage(dist_mat):
    pos = tuple()
    val = 10^5
    for i in range(len(dist_mat)):
        for j in range(i+1, len(dist_mat)):
            elem = dist_mat[i][j]
            if(elem < val and elem > 0):
                val = elem
                pos = (i,j)
    return [pos, val]

## another method for picking element in updated dist_mat  # maaybe not? i don't think so anymore







##the full distance matrix bad thing
## maybe check on the infs/zeros
def update_distance_matrix(dist_mat, pos): #val?
    new_cluster= []
    for x,y in zip(dist_mat[pos[0]], dist_mat[pos[1]]):
        new_cluster.append(x if x > y else y)
     
    dist_mat = np.vstack((dist_mat,new_cluster))
    new_cluster.append(0)
    new_cluster = np.array(new_cluster).reshape(len(new_cluster),1)
    # print(new_cluster) 
    dist_mat = np.hstack((dist_mat,new_cluster))

    #thar be bugs
    
    dist_mat[pos[0]] = [0 for i in dist_mat[pos[0]]]
    dist_mat[pos[1]] = [0 for i in dist_mat[pos[1]]]    
    dist_mat[:,pos[0]] = [0 for i in dist_mat[:,pos[0]]]
    dist_mat[:,pos[1]] = [0 for i in dist_mat[:,pos[1]]]

    #print(new_cluster)
    return dist_mat




def hierarchicalClustering(data):
    
    #will be returned, used for the dendrogram
    #linkageMatrix = []
    
    number_elements_in_cluster = list(np.ones(len(data)))
    
    distanceMatrix = np.zeros((len(data),len(data)))

    #fill distance matrix
    
    for i in range(len(data)):
        for j in range(len(data)): 
            distanceMatrix[i][j] = euclidian_dist(data[i], data[j])
    
    print(distanceMatrix)
    #pick element
    [pos,val] = completeLinkage(distanceMatrix)
    print(pos)
    
    
    
    elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
    #linkageMatrix = np.vstack((linkageMatrix, [pos[0] + 1, pos[1] + 1, val, elements_in_new_cluster])) # pos[0] + 1 and pos[1] + 1?
    linkage_matrix =  [pos[0] , pos[1] , val, elements_in_new_cluster] #+1
    number_elements_in_cluster.append(elements_in_new_cluster)
    
    
    distanceMatrix = update_distance_matrix(distanceMatrix, pos)
    # print(distanceMatrix)
    # print(linkage_matrix)
    while(number_elements_in_cluster[-1] < len(data)):
        [pos,val] = completeLinkage(distanceMatrix)
        elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
        linkage_matrix = np.vstack((linkage_matrix, [pos[0] , pos[1] , val, elements_in_new_cluster])) #+1
        number_elements_in_cluster.append(elements_in_new_cluster)
        distanceMatrix = update_distance_matrix(distanceMatrix, pos)
        # print(pos)
        # print(distanceMatrix)
        # print(linkage_matrix)
    return linkage_matrix

raw_df = pd.read_csv('C:\inf/6\sdm\PA1\P1-Hierarchical-Clustering\Task-1/cc-data.csv')
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

test_data = np.random.rand(20,2)

lm = hierarchicalClustering(test_data)


plt.figure(figsize =(15, 15)) 
plt.title('my') 
Dendrogram = shc.dendrogram(lm)
# plt.title('test') 
# Dendrogram = shc.dendrogram((shc.linkage(test_data, method ='complete', metric = "euclidean")))

#print(lm)
print(lm ==shc.linkage(test_data, method ='complete', metric = "euclidean"))