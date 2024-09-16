from abc import ABC, abstractmethod

from collections import deque

import numpy as np

class Cut():
    def __init__(self, coordinate = 0, threshold = np.inf, left = None, right = None, cluster = -1):
        self.coordinate = coordinate
        self.threshold = threshold
        self.left = left
        self.right = right
        self.cluster = cluster

    def __repr__(self):
        return f"({self.coordinate}, {self.threshold:.2f})"

class Tree():
    def __init__(self):
        self.root = Cut()

    def print_structure(self):
        curr_layer, next_layer = 1, 0
        q = deque()

        q.append(self.root)        
        while q:
            cut = q.popleft()
            
            if cut == None:
                print('N', end=' ')
            else:
                print(cut, end=' ')
                next_layer += 2
                q.append(cut.left)
                q.append(cut.right)

            curr_layer -= 1
            if curr_layer == 0:
                print('')
                curr_layer = next_layer
                next_layer = 0

class ExplainableTree(ABC):
    def __init__(self):
        self.tree = Tree()

    @abstractmethod
    def metric(self, n, l_correct, r_correct, l_centers, r_centers):
        pass

    def single_partition(self, data, clustering, centers, k, present_labels):
        """Decide a single axis-aligned threshold cut for data given clusters and centers.

        Note that data might be some subset of a dataset. Thus, the labeling might refer to more clusters
        then represented by the centers given, and the labeling might not be an arithmetic progression 0,1,2,...
        However, we (hopefully) guarantee that all centers are represented in the data.

        Input: data: np.ndarray - An array of size (number of samples) x (dimension of data)

        clustering: np.ndarray - An array of size (number of samples) that represents the clusters,
        i.e. clustering[j] = i iff x_j in cluster i

        k: int - number of centers (k > 0)

        centers: np.ndarray - An array of size (k) x (dimension of data)

        present_labels: np.ndarray - An array of size (k) that holds labels for the given centers.
        
        Return: 
        best_ratio: float - min_{axis aligned threshold cuts S} phi(S)
        (phi is some function of the cut according to the metric the algo dictates)

        coordinate: int - coordinate of cut

        threshold: int - threshold of cut
        """
        assert k > 1

        if np.intersect1d(clustering, present_labels).size == 0:
            print("Warning: Given centers have no points represented in the data\nNo cuts will be generated further")
            return np.inf, 0, np.NINF

        n = data.shape[0] # num of samples
        assert n > 0
        d = data.shape[1] # dimension of data
        best_ratio = np.inf # Best ratio with respect to metric
        coordinate, threshold = 0, np.NINF 
        u, clusters_count = np.unique(clustering, return_counts=True)
        num_of_labels = u.size # since the tree is build recursively, data separated from it's cluster will not have a cluster present.        
        mapping = dict(zip(u.tolist(), range(num_of_labels))) # since the tree is built recursively, a numbering 0,1,... is not guaranteed.
                                                              # clusters_count[i] = num of samples in cluster mapping[i]
        centerless_count = sum(clusters_count[mapping[c]] for c in u if not c in present_labels)
        
        for i in range(d):
            data_ordering = data[:,i].argsort()
            centers_ordering = centers[:,i].argsort()
            curr_col = data[data_ordering, i]
            clustering_ordered = clustering[data_ordering]
            centers_ordered = centers[centers_ordering]
            indices_ordered = present_labels[centers_ordering]

            centers_threshold = -1 # final center (ordered by i'th coordinate) that is to the left of the cut.
            samples_location = np.zeros(num_of_labels) # samples_location[i] = number of samples that belongs to cluster mapping[i] in left side of cut.
            centerless_left, centerless_right = 0, centerless_count
            
            for j in range(n):
                samples_location[mapping[clustering_ordered[j]]] += 1 # x_j was moved to the left side of the cut.
                if not clustering_ordered[j] in present_labels:
                    centerless_left += 1
                    centerless_right -= 1
                while centers_threshold < k-1 and centers_ordered[centers_threshold+1, i] <= curr_col[j]:
                    centers_threshold += 1
                
                if centers_threshold >= 0:
                    left_side_clusters = [mapping[c] for c in indices_ordered[:centers_threshold+1]] # This sucks
                    l_correct = np.sum(samples_location[left_side_clusters])  
                else:
                    continue

                if centers_threshold < k-1:
                    right_side_clusters = [mapping[c] for c in indices_ordered[centers_threshold+1:]] # This sucks
                    r_correct = np.sum((clusters_count-samples_location)[right_side_clusters])
                else:
                    continue

                l_centers, r_centers = len(left_side_clusters), len(right_side_clusters)

                ratio = self.metric(n, l_correct + centerless_left, r_correct + centerless_right, l_centers, r_centers)
                

                if ratio < best_ratio:
                    coordinate, threshold = i, curr_col[j]
                    best_ratio = ratio
        return best_ratio, coordinate, threshold

    def extend(self, data, clustering, centers, present_labels, cut, indices):
        """Recursively builds the cuts tree.
        
        Input: 
        data: np.ndarray - An array of size (number of samples) x (dimension of data) of the samples that reached the current node.
        clustering: np.ndarray - An array of size (number of samples) of the labeling for data
        centers: np.ndarray - An array of size (number of clusters) x (dimension of data) of the centers that reached the current node.
        present_labels: np.ndarray - An array of size (number of clusters) of the labels of the centers that reached current node.
        cut: Cut - the current node.
        indices: np.ndarray - An array of size (number of clusters) that stores labels for the clusters.
                              This exists to store the clustering of the data according to the tree.
        """
        k = centers.shape[0]
        if k == 1:
            center = np.sum(data, axis=0) / data.shape[0]
            cost = np.sum(np.linalg.norm(data - center, axis=1) ** 2)

            self.cost += cost
            self.centers[self.k, :] = center
            self.clustering[indices] = self.k
            self.k += 1 
            return
        
        best_ratio, coordinate, threshold = self.single_partition(data, clustering, centers, k, present_labels)
        if np.isinf(best_ratio):
            print("Warning: Failed to get a cut.")
            center = np.sum(data, axis=0) / data.shape[0]
            cost = np.sum(np.linalg.norm(data - center, axis=1) ** 2)

            self.cost += cost
            self.centers[self.k, :] = center
            self.clustering[indices] = self.k
            self.k += 1 
            return

        cut.coordinate, cut.threshold = coordinate, threshold
        cut.left, cut.right = Cut(), Cut()

        l_data_partition = data[:,coordinate] <= threshold
        l_data = data[l_data_partition]
        l_clustering = clustering[l_data_partition]
        l_indices = indices[l_data_partition]
        l_centers_partition = centers[:,coordinate] <= threshold
        l_centers = centers[l_centers_partition]
        l_labels = present_labels[l_centers_partition]

        assert l_centers.shape[0] > 0

        self.extend(l_data, l_clustering, l_centers, l_labels, cut.left, l_indices)

        r_data_ordering = data[:,coordinate] > threshold
        r_data = data[r_data_ordering]
        r_clustering = clustering[r_data_ordering]
        r_indices = indices[r_data_ordering]
        r_centers_partition = centers[:,coordinate] > threshold
        r_centers = centers[r_centers_partition]
        r_labels = present_labels[r_centers_partition]

        assert r_centers.shape[0] > 0

        self.extend(r_data, r_clustering, r_centers, r_labels, cut.right, r_indices)

    def train(self, data, clustering, centers):
        """Build the tree of cuts."""
        k = centers.shape[0]
        self.centers = np.zeros_like(centers)
        self.clustering = np.zeros_like(clustering)
        self.k, self.cost = 0, 0
        self.extend(data, clustering, centers, np.arange(k), self.tree.root, np.arange(clustering.size))

class Spectral(ExplainableTree):
    def metric(self, n, l_correct, r_correct, l_centers, r_centers):
        if l_correct == 0 or r_correct == 0:
            return np.inf
        return (n - l_correct - r_correct) / min(l_correct, r_correct)

    
class IMM(ExplainableTree):
    def metric(self, n, l_correct, r_correct, l_centers, r_centers):
        return n - l_correct - r_correct


class EMN(ExplainableTree):
    def metric(self, n, l_correct, r_correct, l_centers, r_centers):
        if l_centers == 0 or r_centers == 0:
            return np.inf
        return (n - l_correct - r_correct) / min(l_centers, r_centers)
