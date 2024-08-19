import numpy as np
from sklearn.neighbors import NearestNeighbors

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable
import heapq

from algo import Tree, Cut

@dataclass(order=True)
class CutComp():
    cond: float 
    cut: Cut = field(compare=False)
    indices: np.ndarray = field(compare=False)

class ExplainableGraphBased():
    def __init__(self):
        self.tree = Tree()

    def single_partition(self, data, graph, current_data):
        # current_data = indices of data points that are in the current node.
        best_cond = np.inf
        best_threshold = np.NINF
        best_index = 0
        dim = data.shape[1]
        n = current_data.size
        #print(f"Data in: {current_data}")

        for i in range(dim):
            data_ordering = current_data[data[current_data,i].argsort()]
            deg_left, deg_right = 0, np.sum(graph[np.ix_(current_data, current_data)]) 
            cut_size = 0

            j = 0
            while j < n-1:
                threshold = data[data_ordering[j],i]
                while j < n-1 and data[data_ordering[j], i] == threshold:
                    left_weight = np.sum(graph[data_ordering[j], data_ordering[:j]])
                    right_weight = np.sum(graph[data_ordering[j], data_ordering[j+1:]])

                    cut_size += right_weight - left_weight
                    deg_left += left_weight * 2
                    deg_right -= right_weight * 2

                    j += 1

                if deg_left == 0 or deg_right == 0:
                    continue

                cond = cut_size / min(deg_left, deg_right)
                if cond < best_cond:
                    best_cond = cond
                    best_threshold = threshold
                    best_index = i

        #print(f"Res - Cond - {best_cond}, Index - {best_index}, Treshold - {best_threshold}")
        #print(current_data)
        return best_cond, best_index, best_threshold

    def train(self, data, graph, k):
        """Build the tree of cuts."""
        #self.extend(data, graph, np.arange(data.shape[0]), self.tree.root)

        curr_k = 1

        heap = []
        current_data = np.arange(data.shape[0])
        best_cond, best_index, best_threshold = self.single_partition(data, graph, current_data)
        self.tree.root.coordinate = best_index
        self.tree.root.threshold = best_threshold
        heapq.heappush(heap, CutComp(best_cond, self.tree.root, current_data))

        while curr_k < k:
            node = heapq.heappop(heap)
            cut, current_data = node.cut, node.indices
            left_bool = data[current_data,cut.coordinate] <= cut.threshold
            l_partition = current_data[left_bool]
            r_partition = current_data[~left_bool]

            l_best_cond, l_best_index, l_best_threshold = self.single_partition(data, graph, l_partition)
            r_best_cond, r_best_index, r_best_threshold = self.single_partition(data, graph, r_partition)

            cut_left = Cut(l_best_index, l_best_threshold)
            cut_right = Cut(r_best_index, r_best_threshold)

            cut.left = cut_left
            cut.right = cut_right

            heapq.heappush(heap, CutComp(l_best_cond, cut_left, l_partition))
            heapq.heappush(heap, CutComp(r_best_cond, cut_right, r_partition))

            curr_k += 1

        while heap:
            heapq.heappop(heap).cut.cluster = curr_k
            #print(curr_k)
            curr_k -= 1

        assert curr_k == 0

    def predict(self, data):
        current_data = np.arange(data.shape[0])
        queue = [(self.tree.root, current_data)]
        clustering = np.zeros(data.shape[0])
        while queue:
            cut, current_data = queue.pop()
            if cut.left == None:
                clustering[current_data] = cut.cluster
                #print(cut.cluster)
            else:
                left_bool = data[current_data,cut.coordinate] <= cut.threshold
                l_partition = current_data[left_bool]
                r_partition = current_data[~left_bool]
                #print(f"Coordinate - {cut.coordinate}, Threshold {cut.threshold}")
                #print(f"L - {l_partition}")
                #print(f"R - {r_partition}")
                if l_partition.size != 0:
                    queue.append((cut.left, l_partition))
                if r_partition.size != 0:
                    queue.append((cut.right, r_partition))

        return clustering


def get_nearest_neighbors(data, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)

    mat = nbrs.kneighbors_graph() # This is a sparse CSR matrix. do toarray() if something goes wrong???
    return mat + mat.T
        
