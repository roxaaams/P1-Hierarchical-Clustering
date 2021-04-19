

from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt

def print_scipy_solution(a):
    professional_implementation = shc.linkage(
        a, method='complete', metric="euclidean")
    print(professional_implementation)
    #tests = [(x,y) for i in range(len(professional_implementation)) for x,y in zip(result[i],professional_implementation[i]) if x != y ]
    # sch.dendrogram
    # print(tests)
    #print(all([all(x == y) for x,y in zip(result,professional_implementation)]))

    linked = linkage(a, 'complete')
    labelList = range(0, 10)
    #plt.figure(figsize=(10, 7))
    # shc.dendrogram(linked,)
    # plt.show()