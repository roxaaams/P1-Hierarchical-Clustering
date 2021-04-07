# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 00:19:10 2021

@author: pault
"""

from joblib import Parallel, delayed
import numpy as np
import time
import multiprocessing

print(multiprocessing.cpu_count())

def euclid_np(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def euclid(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** (1/2)


data = np.random.rand(700,2)


for i in [-1,8]: #range(4,multiprocessing.cpu_count()+1,2):
    jobs = i
    print(jobs, "\n")
    
    
    # ## euclid_np
    # print("euclid_np \n")
    # # brute
    
    # dm_brute = np.zeros((len(data),len(data)))
    
    # start = time.time()
    
    # for i in range(len(data)):
    #     for j in range(len(data)):
    #         dm_brute[i][j] = euclid_np(data[i], data[j])
    
    # stop = time.time()
    
    # print("dm_brute: ", stop - start)
    
    # # joblib cpu
    
    # stack = False
    
    # if(stack):
        
    #     start = time.time()
        
    #     dm_joblib_cp = Parallel(n_jobs = jobs)(delayed(euclid_np)(data[0],data[i]) for i in range(len(data)))
        
    #     for i in range(1,len(data)):
    #         dm_joblib_cp = np.vstack((dm_joblib_cp, Parallel(n_jobs = jobs)(delayed(euclid_np)(data[i],data[j]) for j in range(len(data)))))
    
    #     stop = time.time()
    
    # else:
    #     start = time.time()
    #     dm_joblib_cp = Parallel(n_jobs = jobs)(delayed(euclid_np)(data[i],data[j]) for i in range(len(data)) for j in range(len(data)))
    #     stop = time.time()
        
        
    # print("dm_joblib_cp: ", stop - start)
    
    
    # #joblib thread
    
    # stack = False
    
    # if(stack):
        
    #     start = time.time()
        
    #     dm_joblib_thr = Parallel(n_jobs = jobs, prefer = "threads")(delayed(euclid_np)(data[0],data[i]) for i in range(len(data)))
        
    #     for i in range(1,len(data)):
    #         dm_joblib_thr = np.vstack((dm_joblib_thr, Parallel(n_jobs = jobs,prefer = "threads")(delayed(euclid_np)(data[i],data[j]) for j in range(len(data)))))
    
    #     stop = time.time()
    
    # else:
    #     start = time.time()
    #     dm_joblib_thr = Parallel(n_jobs = jobs, prefer = "threads")(delayed(euclid_np)(data[i],data[j]) for i in range(len(data)) for j in range(len(data)))
    #     stop = time.time()
        
        
    # print("dm_joblib_thr: ", stop - start)
    
    
    # ##
    # print("\n")
    # ##
    
    
    
    
    
    ##euclid
    
    print("euclid \n")
    # brute
    
    dm_brute = np.zeros((len(data),len(data)))
    
    start = time.time()
    
    for i in range(len(data)):
        for j in range(len(data)):
            dm_brute[i][j] = euclid(data[i], data[j])
    
    stop = time.time()
    
    print("dm_brute: ", stop - start)
    
    # joblib cpu
    
    stack = False
    
    if(stack):
        
        start = time.time()
        
        dm_joblib_cp = Parallel(n_jobs = jobs)(delayed(euclid)(data[0],data[i]) for i in range(len(data)))
        
        for i in range(1,len(data)):
            dm_joblib_cp = np.vstack((dm_joblib_cp, Parallel(n_jobs = jobs)(delayed(euclid)(data[i],data[j]) for j in range(len(data)))))
    
        stop = time.time()
    
    else:
        start = time.time()
        dm_joblib_cp = Parallel(n_jobs = jobs)(delayed(euclid)(data[i],data[j]) for i in range(len(data)) for j in range(len(data)))
        stop = time.time()
        
        
    print("dm_joblib_cp: ", stop - start)
    
    
    #joblib thread
    
    stack = False
    
    if(stack):
        
        start = time.time()
        
        dm_joblib_thr = Parallel(n_jobs = jobs, prefer = "threads")(delayed(euclid)(data[0],data[i]) for i in range(len(data)))
        
        for i in range(1,len(data)):
            dm_joblib_thr = np.vstack((dm_joblib_cp, Parallel(n_jobs = jobs,prefer = "threads")(delayed(euclid)(data[i],data[j]) for j in range(len(data)))))
    
        stop = time.time()
    
    else:
        start = time.time()
        dm_joblib_thr = Parallel(n_jobs = jobs, prefer = "threads")(delayed(euclid)(data[i],data[j]) for i in range(len(data)) for j in range(len(data)))
        stop = time.time()
        
        
    print("dm_joblib_thr: ", stop - start)


    ##
    print("\n")
    ##


