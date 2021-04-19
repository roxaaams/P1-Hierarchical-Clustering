import numpy as np
import sys

from collections import deque
from sklearn.metrics import pairwise_distances as pair_dist

from preprocessing import preprocess
from distance_computation import update_distance_matrix, complete_linkage
from scipy_solution import print_scipy_solution

np.set_printoptions(threshold=sys.maxsize, precision=2)

a = preprocess()
a = a[:10]
cluster_index = len(a)
clusters = []
global distance_matrix
distance_matrix = pair_dist(a)

def nearest_neighbor():
    global cluster_index
    global distance_matrix
    global a

    # active clustsers are saved as indices to points in a
    active_clusters = [i for i in range(0, len(a))]
    clusters = []

    # S is a stack
    S = deque()

    while len(active_clusters) >= 1:
        distances = []

        # S structure:
        # consists of 2-value arrays where the first value is the cluster index,
        # and the second value is the cluster

        # change clusters saved in S to contain only indices

        # if stack is empty, append the first active cluster
        if not S:
            S.append([active_clusters[0], [active_clusters[0]]])
            active_clusters.remove(active_clusters[0])
        # when a cluster is pushed to the stack, delete it in active_clusters
        elif len(S) == 1:
            unpacked = unpack_cluster(S[-1])
            nearest_distance, cluster, is_stack = find_nearest(
                active_clusters, unpacked)
            S.append(cluster)

            if cluster in active_clusters:
                active_clusters.remove(cluster)
            elif cluster[0] in active_clusters:
                active_clusters.remove(cluster[0])
        else:
            unpacked_1 = unpack_cluster(S[-1])
            unpacked_2 = unpack_cluster(S[-2])
            nearest_distance, cluster, is_stack = find_nearest(
                active_clusters, unpacked_1, unpacked_2)
            if is_stack:
                cluster = S[-1][-2]

                predecessor_cluster = S.pop()
                comparable_cluster = S.pop()

                merged_cluster = merge_clusters(
                    cluster_index, predecessor_cluster, comparable_cluster)

                clusters.append({"distance": nearest_distance, "cluster": [
                                cluster_index, [predecessor_cluster[0], comparable_cluster[0]]]})

                # add predecessor and comparable cluster index to clusters instead
                pos = sorted([min(predecessor_cluster[1]),
                             min(comparable_cluster[1])])

                distance_matrix = update_distance_matrix(
                    distance_matrix, pos, a)

                active_clusters.append(merged_cluster)

                cluster_index += 1
            else:
                S.append(cluster)
                if cluster in active_clusters:
                    active_clusters.remove(cluster)
                elif cluster[0] in active_clusters:
                    active_clusters.remove(cluster[0])

    while len(S) > 1:
        pred = S.pop()
        preped = S.pop()

        unpacked_1 = unpack_cluster(pred)
        unpacked_2 = unpack_cluster(preped)

        nearest_distance, cluster, is_stack = find_nearest(
            active_clusters, unpacked_1, unpacked_2)

        merged = merge_clusters(cluster_index, pred, preped)

        clusters.append({"distance": nearest_distance, "cluster": [
                        cluster_index, [pred[0], preped[0]]]})
        S.append(merged)

        cluster_index += 1

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

# find nearest cluster
def find_nearest(active, cluster, stack_pred=None):
    if isinstance(cluster, int):
        cluster = [cluster]

    nearest_distance = -1
    nearest_cluster = []
    is_stack = False

    for i in active:
        aggdiv = None
        nrst = i

        if isinstance(i, int):
            aggdiv = [i]
            nrst = i
        else:
            aggdiv = unpack_cluster(i)
            nrst = i

        temp_distance = complete_linkage(aggdiv, cluster, distance_matrix)

        if nearest_distance < 0 or nearest_distance > temp_distance:
            nearest_distance = temp_distance
            nearest_cluster = nrst

    if stack_pred != None:
        if isinstance(stack_pred, int):
            stack_pred = [stack_pred]

        temp_distance = complete_linkage(stack_pred, cluster, distance_matrix)

        if temp_distance <= nearest_distance or nearest_distance < 0:
            is_stack = True
            nearest_distance = temp_distance

    if isinstance(nearest_cluster, int):
        return [nearest_distance, [nearest_cluster, [nearest_cluster]], is_stack]

    return [nearest_distance, nearest_cluster, is_stack]


def merge_clusters(pos, c1, c2):
    merged_cluster = []

    c1 = unpack_cluster(c1)
    c2 = unpack_cluster(c2)

    merged_cluster.extend(c1)
    merged_cluster.extend(c2)

    return [pos, merged_cluster]

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

# for line in distance_matrix:
#    print(line)
nearest_neighbor()
print_scipy_solution(a)
