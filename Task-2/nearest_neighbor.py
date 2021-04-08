import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

#from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
#from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
import sys

from collections import deque


np.set_printoptions(threshold=sys.maxsize)



'''

	Initialize the set of active clusters to consist of n one-point clusters, one for each input point.
	Let S be a stack data structure, initially empty, the elements of which will be active clusters.
	While there is more than one cluster in the set of clusters:
		If S is empty, choose an active cluster arbitrarily and push it onto S.
		Let C be the active cluster on the top of S. Compute the distances from C to all other clusters, and let D be the nearest other cluster.
		If D is already in S, it must be the immediate predecessor of C. Pop both clusters from S and merge them.
		Otherwise, if D is not already in S, push it onto S.


'''

clusters = []

def nearest_neighbor(a):
	active_clusters = a
	distances = []

	S = deque(active_clusters[0])

	while len(active_clusters) > 1:
		distances = []
		if not S:
			S = deque(active_clusters[0])
		nearest_distance, cluster = find_nearest(active_clusters, S[-1] if len(S) == 1 else active_clusters[0])
		np.delete(active_clusters, np.where(active_clusters == S[-1]))
		if cluster.all() in S:
			predecessor_cluster = S.pop()
			comparable_cluster = S.pop()
			clusters.append({ distance: nearest_distance, cluster: cluster})
			predecessor_cluster.extend(comparable_cluster)
			S.append(predecessor_cluster)
		else:
			S.append(cluster)


def euclidean_norm(points):

	res = 0
	if isinstance(points[0], int):
		return np.sqrt((points[0]-points[1])**2)

	res += (points[0]-points[1])**2

	return np.sqrt(res)

def complete_linkage(cluster1, cluster2):

	distances = []
	print("cluster1", cluster1)
	print("cluster2", cluster2)


	for point_x in cluster1:
		#print("cluster1", cluster1)
		for point_y in cluster2:
			#print("cluster2", cluster2)
			distances.append(euclidean_norm([point_x, point_y]))

	return max(distances)

def find_nearest(data, cluster):
	l = len(data)

	nearest_distance = 0
	nearest_cluster = []
	for i in range(0, l):
		if nearest_distance == 0:
			nearest_distance = complete_linkage(data[i], cluster)
		elif nearest_distance > complete_linkage(data[i], cluster):
			nearest_distance = complete_linkage(data[i], cluster)
		nearest_cluster = data[i]
	#print(data[nearest_distance, data[nearest_distance]])
	print(nearest_distance, nearest_cluster)
	return [ nearest_distance, nearest_cluster ]

# read in data
raw_df = pd.read_csv('CC GENERAL.csv')
raw_df = raw_df.drop('CUST_ID', axis = 1) 
raw_df.fillna(method ='ffill', inplace = True) 
raw_df.head(2)

# Standardize data
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(raw_df) 
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df) 
  
# Reducing the dimensions of the data 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
  
X_principal.head(2)

# convert dataFrame to list
a = np.array(np.array(X_principal.values.tolist()))
#print(a)

nearest_neighbor(a)
	

