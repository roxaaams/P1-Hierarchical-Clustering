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

cluster_index = len(a)
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

all_distances = []



# rewrite this

def merge_clusters(index, c1, c2):

	merged_cluster = []

	c1 = unpack_cluster(c1)
	c2 = unpack_cluster(c2)

	merged_cluster.extend(c1)
	merged_cluster.extend(c2)

	return [index, merged_cluster]


def unpack_cluster(cluster):
	unpacked_cluster = []
	if isinstance(cluster[0], int) and isinstance(cluster[-1][0], int):
		return cluster[-1]
	elif isinstance(cluster[0], int):
		for val in cluster[1]:
			unpacked_cluster.append(val[-1][0])
	else:
		for val in cluster:
			unpacked_cluster.append(val[-1][0])
	return unpacked_cluster


def nearest_neighbor():
	global cluster_index

	# active clustsers are saved as indices to points in a
	active_clusters = [i for i in range(0, len(a))]
	active_clusters = active_clusters[::-1]
	distances = []
	clusters = []

	# todo: pre-compute distances between all points (nxn)
	# todo: save clusters as scipy linkage format

	# S is a stack
	S = deque()

	while len(active_clusters) >= 1:
		print("S every iterations", S)
		distances = []

		# S structure:
		# consists of 2-value arrays where the first value is the cluster index, 
		# and the second value is the cluster


		# change clusters saved in S to contain only indices

		# if stack is empty, append the first active cluster
		if not S:
			S.append([active_clusters[0], [active_clusters[0]]]) # put val in braces?
			active_clusters.remove(active_clusters[0])

		# when a cluster is pushed to the stack, delete it in active_clusters

		if len(S) <= 1:
			unpacked = unpack_cluster(S[-1])
			nearest_distance, cluster, is_stack = find_nearest(active_clusters, unpacked)
			S.append([cluster,  [cluster]])
			clusters.append({ "distance": nearest_distance, "cluster": cluster})
			active_clusters.remove(cluster)

		else:
			unpacked_1 = unpack_cluster(S[-1])
			unpacked_2 = unpack_cluster(S[-2])
			nearest_distance, cluster, is_stack = find_nearest(active_clusters, unpacked_1, unpacked_2)
			if is_stack:
				cluster = S[-1][-2]

				predecessor_cluster = S.pop()
				comparable_cluster = S.pop()

				merged_cluster = merge_clusters(cluster_index, predecessor_cluster, comparable_cluster)

				#clusters.append({ "distance": nearest_distance, "cluster": merged_cluster})
				clusters.append({ "distance": nearest_distance, "cluster": [cluster_index,[predecessor_cluster[0], comparable_cluster[0]]]})

				# add predecessor and comparable cluster index to clusters instead
				S.append(merged_cluster)
				cluster_index += 1
			else:
				print(cluster)
				S.append([cluster, [cluster]])
				clusters.append({ "distance": nearest_distance, "cluster": cluster})
				# add predecessor and comparable cluster index to clusters instead
				active_clusters.remove(cluster)


	# combine remaining clusters on the stack
	while len(S) >  1:

		unpacked_1 = unpack_cluster(S[-1])
		unpacked_2 = unpack_cluster(S[-2])

		nearest_distance, cluster, is_stack = find_nearest(active_clusters, unpacked_1, unpacked_2)

		if is_stack:
			cluster = S[-1][-2]

			predecessor_cluster = S.pop()
			comparable_cluster = S.pop()

			merged_cluster = merge_clusters(cluster_index, predecessor_cluster, comparable_cluster)

			#clusters.append({ "distance": nearest_distance, "cluster": merged_cluster})
			clusters.append({ "distance": nearest_distance, "cluster": [cluster_index,[predecessor_cluster[0], comparable_cluster[0]]]})

			# add predecessor and comparable cluster index to clusters instead

			S.append(merged_cluster)
			cluster_index += 1
	
	# print all clusters for debugging
			
	#print(clusters)
	for i in clusters:
		print(i)

	'''
	line =  "[		{c1}.		{c2}.		{distance}			{nr}.		]"
	total_number_elements_per_cluster = []
	print("[")
	for cl in clusters:
		if (not isinstance(cl.cluster, int)):
			total_number_elements_per_cluster.append(len(cl.cluster))
		print(line.format( , , , ))
   '''


# calculate the euclidean norm of two points
def euclidean_norm(points, dimensions=2):

	res = 0

	for component in range(0, dimensions):
		res += (a[points[0]][component]-a[points[1]][component])**2

	return np.sqrt(res)


def complete_linkage_old(cluster1, cluster2):

	distances = []

	for point_x in cluster1:
		for point_y in cluster2:
			distances.append(euclidean_norm([point_x, point_y]))

	return max(distances)

# find the distance between two clusters with the complete linkage method
# todo: optimize out complete linkage by saving distances between clusters
def complete_linkage(cluster1, cluster2):

	distances = []

	for point_x in cluster1:
		for point_y in cluster2:
			distances.append(all_distances[point_x][point_y])

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



def euclidean_norm_numeric(points, dimensions=2):

	res = 0

	for component in range(0, dimensions):
		res += (points[0][component]-points[1][component])**2

	return np.sqrt(res)

all_distances = [[euclidean_norm_numeric([i, j]) for j in a] for i in a]
#all_distances = [[[i,j] for j in range(0, len(a))] for i in range(0, len(a))]

for line in all_distances:
	print(line)

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

