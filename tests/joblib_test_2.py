# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 02:27:09 2021

@author: pault
"""

from joblib import Parallel, delayed
import numpy as np
import time
import multiprocessing
from sklearn.metrics import pairwise_distances as pair_dist
# def euclid(a,b):
#     dist = 0
#     for i in range(len(a)):
#         dist += (a[i] - b[i]) ** 2
#     return dist ** (1/2)

# def dist_mat(data):
#     dm = np.zeros((len(data),len(data)))
#     for i in range(len(data)):
#         for j in range(len(data)):
#             dm[i][j] = euclid(data[i], data[j])
#     return dm
            
            
# data = np.random.rand(10,2)
# jobs = -1


# dm_brute = np.zeros((len(data),len(data)))
# start = time.time()

# for i in range(len(data)):
#     for j in range(len(data)):
#         dm_brute[i][j] = euclid(data[i], data[j])

# stop = time.time()

# print("dm_brute: ", stop - start)

# # joblib cpu


# prolist = [i for i in range(0,len(data), 2)]
# start = time.time()

# dm_joblib = Parallel(n_jobs = jobs)(delayed(dist_mat)(data[prolist[i-1]:prolist[i]]) for i in range(1,len(prolist))) 
# stop = time.time()

    
# print("dm_joblib_cp: ", stop - start)

# finalize = dm_joblib[0]
# for i in range(1, len(dm_joblib)):
#     finalize = np.vstack((finalize,dm_joblib[i]))

# print(finalize)




# def euclid(a,b):
#     dist = 0
#     for i in range(len(a)):
#         dist += (a[i] - b[i]) ** 2
#     return dist ** (1/2)

# def dist_mat(data,vert_start,vert_end, horz):
#     dm = np.zeros((vert_end - vert_start,horz))

#     for i in range(len(dm)):
#         for j in range(horz):
#             dm[i][j] = euclid(data[i], data[j])
#     return dm


# data = np.random.rand(4,2)
# jobs = -1


# dm_brute = np.zeros((len(data),len(data)))
# start = time.time()

# for i in range(len(data)):
#     for j in range(len(data)):
#         dm_brute[i][j] = euclid(data[i], data[j])

# stop = time.time()

# print("dm_brute: ", stop - start)

# # joblib cpu


# prolist = [i for i in range(0, len(data)+1, 2)]
# start = time.time()

# dm_joblib = Parallel(n_jobs = jobs)(delayed(dist_mat)(data, prolist[i-1], prolist[i], len(data)) for i in range(1,len(prolist))) 
# stop = time.time()


    
# print("dm_joblib_cp: ", stop - start)

# finalize = dm_joblib[0]
# for i in range(1, len(dm_joblib)):
#     finalize = np.vstack((finalize,dm_joblib[i]))


# print(dm_brute)
# print(finalize)
# print(prolist)

# print(dm_joblib)



def euclid(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** (1/2)

def dist_mat(data,vert_start,vert_end, horz):
    dm = np.zeros((vert_end - vert_start,horz))

    for i in range(len(dm)):
        for j in range(horz):
            dm[i][j] = euclid(data[i+vert_start], data[j])
    return dm


def create_split_indices(data, size = 100):
    split = len(data)//size
    print(split)
    lengths = [i for i in range (0, len(data)+1, split)] ##!!! if split != 0, not included for performance
    lengths.append(len(data)) if lengths[-1] != len(data) else None
    
    return lengths
    
    
data = np.random.rand(2000,2)
jobs = -1


# dm_brute = np.zeros((len(data),len(data)))
# start = time.time()

# for i in range(len(data)):
#     for j in range(len(data)):
#         dm_brute[i][j] = euclid(data[i], data[j])

# stop = time.time()

# print("dm_brute: ", stop - start)

# joblib cpu


prolist = [i for i in range(0, len(data)+1, 100)]
start = time.time()

dm_joblib = Parallel(n_jobs = jobs)(delayed(dist_mat)(data, prolist[i-1], prolist[i], len(data)) for i in range(1,len(prolist))) 
stop = time.time()


    
print("dm_joblib_cp: ", stop - start)

finalize = dm_joblib[0]
for i in range(1, len(dm_joblib)):
    finalize = np.vstack((finalize,dm_joblib[i]))


start = time.time()
distanceMatrix = pair_dist(data), n_jobs = jobs)
stop = time.time()

print("godlike: ", stop - start)
# print(dm_brute)
# print(finalize)
print(prolist)
print(finalize)
# print(dm_joblib)
print(distanceMatrix)

print(prolist == create_split_indices(data))

prolist_data = np.random.rand(123,2)

print(create_split_indices(prolist_data))