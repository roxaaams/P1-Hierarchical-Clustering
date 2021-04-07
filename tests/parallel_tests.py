# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 20:31:00 2021

@author: pault
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

import multiprocessing
from joblib import Parallel, delayed

import time

from sklearn.metrics import pairwise_distances as m_pd

#def: 

def euclidian_dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

### THIS SECTION TAKEN FROM THE KAGGLE NOTEBOOK

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
data = pca.fit_transform(normalized_df) 

data = np.random.rand(3,2)
#using cpus

# p_cpu_start = time.time()
# #p_cpu = Parallel(n_jobs = -1)(delayed(euclidian_dist)(data[i],data[j]) for i in range(len(data)) for j in range(len(data)))  
# p_cpu = np.zeros(len(data))
# for i in range(len(data)):
#     p_cpu = np.vstack((p_cpu, Parallel(n_jobs = -1)(delayed(euclidian_dist)(data[i],data[j]) for j in range(len(data)))))

# p_cpu_end = time.time()

# print("p_cpu: ", p_cpu_end - p_cpu_start )

# #using threads

# p_thr_start = time.time()
# #p_thr =  Parallel(n_jobs = -1, prefer = "threads")(delayed(euclidian_dist)(data[i],data[j]) for i in range(len(data)) for j in range(len(data)))
# p_thr = np.zeros(len(data))
# for i in range(len(data)):
#     p_thr = np.vstack((p_thr, Parallel(n_jobs = -1, prefer = "threads")(delayed(euclidian_dist)(data[i],data[j]) for j in range(len(data)))))

# p_thr_end = time.time()

# print("p_thr: ", p_thr_end - p_thr_start)

#using pairwise_distance

p_pd_start = time.time()
p_pd = m_pd(data, n_jobs = -1)
p_pd_end = time.time()

print("p_pd: ", p_pd_end - p_pd_start )

#using good old brute

p_dm_start = time.time()

dm = np.zeros((len(data),len(data)))
for i in range(len(data)):
    for j in range(len(data)):
        dm[i][j] = euclidian_dist(data[i],data[j])



p_dm_end = time.time()

print("p_dm: ", p_dm_end - p_dm_start )
    
# print("cpu, dm", p_cpu == dm)
# print("thr, dm", p_thr == dm)
print("pd, dm", (p_pd == dm).all())

# print(p_cpu)
# print(p_thr)
print(p_pd)
print(dm)
no_same = [(p_pd[i][j], dm[i][j]) for i in range(len(data)) for j in range(len(data)) if p_pd[i][j] != dm[i][j]]
for i in no_same:
    print(i)