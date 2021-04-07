# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:40:30 2021

@author: pault
"""
import numpy as np

test = [[1,2],[2,3]]

def change(test):
    test[1][0] = 0
    

change(test)
print(test)


yolo = [5] * 10
print(yolo)

yolt = np.ones(len(yolo))
print(yolt)
print(yolo == yolt*5)
print(list(yolt)*5)
[a,b] = [2,3]
print(b)
c = (1,2)
print(c[0])



cd = [1,2,3,4]
bf = [4,5,6,7]
yolthr = []
for x,y in zip(cd,bf) :
    yolthr.append(x if x > y else y)
print(yolthr)

def update_distance_matrix(dist_mat, pos, val): 
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


def euclidian_dist(a,b):
    return np.sum(np.square(a-b)) ** (1/2)







#testingData = np.array([1,2],[2,3],[3,4],[4,5],[5,6])
data = np.array([[1,2],[3,4],[5,6]])
newarr = np.array(np.split(np.arange(1,7),3))
print(data == newarr)




distanceMatrix = np.empty((len(data),len(data)))


for i in range(len(data)):
    for j in range(len(data)):
        distanceMatrix[i][j] = euclidian_dist(data[i], data[j])
print(distanceMatrix)

update_distance_matrix(distanceMatrix, (1,2), 3)

print(distanceMatrix)






















