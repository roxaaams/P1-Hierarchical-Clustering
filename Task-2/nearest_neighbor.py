import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

#from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
#from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
import sys

from matplotlib import pyplot as plt

from collections import deque


np.set_printoptions(threshold=sys.maxsize)



# reduce dimensions of data

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
a = X_principal.values.tolist()

a = a[:10]


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


def merge_clusters(c1, c2):

	merged_cluster = []
	if isinstance(c1, int):
		merged_cluster.append(c1)
	else:
		merged_cluster.extend(c1)
	if isinstance(c2, int):
		merged_cluster.append(c2)
	else:
		merged_cluster.extend(c2)
	return merged_cluster


def nearest_neighbor():

	# active clustsers are saved as indices to points in a
	active_clusters = [i for i in range(0, len(a))]

	distances = []
	clusters = []

	# todo: pre-compute distances between all points (nxn)
	# todo: save clusters as scipy linkage format

	# S is a stack
	S = deque()


	while len(active_clusters) >= 1:
		print("S every iterations", S)
		distances = []

		# if stack is empty, append the first active cluster
		if not S:
			S = deque([active_clusters[0]])
			active_clusters.remove(S[-1])


		# when a cluster is pushed to the stack, delete it in active_clusters

		if len(S) <= 1:
			nearest_distance, cluster, is_stack = find_nearest(active_clusters, S[-1])
			S.append(cluster)
			clusters.append({ "distance": nearest_distance, "cluster": cluster})
			active_clusters.remove(cluster)
		else:
			nearest_distance, cluster, is_stack = find_nearest(active_clusters, S[-1], S[-2])
			if is_stack:
				cluster = S[-2]

				predecessor_cluster = S.pop()
				comparable_cluster = S.pop()

				merged_cluster = merge_clusters(predecessor_cluster, comparable_cluster)

				clusters.append({ "distance": nearest_distance, "cluster": merged_cluster})
				S.append(merged_cluster)
			else:
				S.append(cluster)
				clusters.append({ "distance": nearest_distance, "cluster": cluster})
				active_clusters.remove(cluster)

		print(S)

	# combine remaining clusters on the stack
	while len(S) >  1:

		nearest_distance, cluster, is_stack = find_nearest(active_clusters, S[-1], S[-2])

		if is_stack:
			print("entered the if")
			cluster = S[-2]

			predecessor_cluster = S.pop()
			comparable_cluster = S.pop()

			merged_cluster = merge_clusters(predecessor_cluster, comparable_cluster)

			clusters.append({ "distance": nearest_distance, "cluster": merged_cluster})
			S.append(merged_cluster)
	
	# print all clusters for debugging		
	print(clusters)


# calculate the euclidean norm of two points
def euclidean_norm(points, dimensions=2):

	res = 0

	for component in range(0, dimensions):
		res += (a[points[0]][component]-a[points[1]][component])**2

	return np.sqrt(res)

# find the distance between two clusters with the complete linkage method
def complete_linkage(cluster1, cluster2):

	distances = []

	for point_x in cluster1:
		for point_y in cluster2:
			distances.append(euclidean_norm([point_x, point_y]))

	return max(distances)

# find nearest cluster
def find_nearest(active, cluster, stack_pred=None):

	if isinstance(cluster, int):
		cluster = [cluster]

	nearest_distance = -1
	nearest_cluster = []
	is_stack = False

	for i in active:

		temp_distance = complete_linkage([i], cluster)

		if nearest_distance < 0 or nearest_distance > temp_distance:
			nearest_distance = temp_distance
			nearest_cluster = i

	if stack_pred != None:

		if isinstance(stack_pred, int):
			stack_pred = [stack_pred]

		temp_distance = complete_linkage(stack_pred, cluster)

		if temp_distance <= nearest_distance or nearest_distance < 0:
			is_stack = True
			nearest_distance = temp_distance

	return [ nearest_distance, nearest_cluster, is_stack ]



nearest_neighbor()

professional_implementation = shc.linkage(a, method ='complete', metric = "euclidean")
print(professional_implementation)
#tests = [(x,y) for i in range(len(professional_implementation)) for x,y in zip(result[i],professional_implementation[i]) if x != y ]

#sch.dendrogram
#print(tests)
#print(all([all(x == y) for x,y in zip(result,professional_implementation)]))

linked = linkage(a, 'complete')

labelList = range(0, 10)

#plt.figure(figsize=(10, 7))
#shc.dendrogram(linked,)
#plt.show()

