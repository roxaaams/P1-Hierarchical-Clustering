# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:37:37 2021

@author: pault
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from scipy.spatial import distance_matrix

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA


def euclidian_dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))


# def completeLinkage(distMat): 
#     all_row_pos = tuple()
#     all_row_val = 10^5
#     for i in range(len(distMat)):
#         one_row_val = -1
#         one_row_pos = tuple()
#         for j in range(i+1,len(distMat)):   ##There will be an error because out of range (just treat edge-cases)
#             elem = distMat[i][j]
#             if(elem >= one_row_val):
#                 one_row_val = elem
#                 one_row_pos = (i,j)
#         if(one_row_val <= all_row_val and one_row_val >= 0):
#             print(one_row_pos,one_row_val)
#             all_row_val = one_row_val
#             all_row_pos = one_row_pos
    
#     return [all_row_pos, all_row_val]


# data = np.random.rand(10,2)
# print(data)
# distanceMatrix = np.empty((len(data),len(data)))

# for i in range(len(data)):
#     for j in range(len(data)):
#         distanceMatrix[i][j] = euclidian_dist(data[i], data[j])

# print(distanceMatrix)
# number_elements_in_cluster = list(np.ones(len(data)))

# [pos,val] = completeLinkage(distanceMatrix)


# elements_in_new_cluster = number_elements_in_cluster[pos[0]] + number_elements_in_cluster[pos[1]]
# number_elements_in_cluster.append(elements_in_new_cluster)

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

distanceMatrix = np.empty((len(test_data),len(test_data)))


distance_matrix(test_data, test_data)

print("check")
for i in range(len(test_data)):
    for j in range(i+1,len(test_data)):
        distanceMatrix[i][j] = euclidian_dist(test_data[i], test_data[j])
print("check")