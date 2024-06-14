from abc import ABC, abstractmethod

import numpy as np

class Cut():
    def __init__(self, coordinate = 0, threshold = np.inf, left = None, right = None):
        self.coordinate = coordinate
        self.threshold = threshold
        self.left = left
        self.right = right

class Tree():
    def __init__(self):
        self.root = Cut()

class ExplainableTree(ABC):
    def __init__(self):
        self.tree = Tree()

    @abstractmethod
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
        pass

    def extend(self, data, clustering, centers, present_labels, cut):
        k = centers.shape[0]
        if k == 1:
            center = np.sum(data, axis=0) / data.shape[0]
            cost = np.sum(np.linalg.norm(data - center, axis=1) ** 2)

            self.cost += cost
            self.centers[self.k, :] = center
            self.k += 1 
            return
        
        _, coordinate, threshold = self.single_partition(data, clustering, centers, k, present_labels)
        cut.coordinate, cut.threshold = coordinate, threshold
        cut.left, cut.right = Cut(), Cut()

        l_data_partition = data[:,coordinate] <= threshold
        l_data = data[l_data_partition]
        l_clustering = clustering[l_data_partition]
        l_centers_partition = centers[:,coordinate] <= threshold
        l_centers = centers[l_centers_partition]
        l_labels = present_labels[l_centers_partition]

        assert l_centers.shape[0] > 0

        self.extend(l_data, l_clustering, l_centers, l_labels, cut.left)

        r_data_ordering = data[:,coordinate] > threshold
        r_data = data[r_data_ordering]
        r_clustering = clustering[r_data_ordering]
        r_centers_partition = centers[:,coordinate] > threshold
        r_centers = centers[r_centers_partition]
        r_labels = present_labels[r_centers_partition]

        assert r_centers.shape[0] > 0

        self.extend(r_data, r_clustering, r_centers, r_labels, cut.right)

    def train(self, data, clustering, centers):
        k = centers.shape[0]
        self.centers = np.zeros_like(centers)
        self.k, self.cost = 0, 0
        self.extend(data, clustering, centers, np.arange(k), self.tree.root)

class Spectral(ExplainableTree):
    def single_partition(self, data, clustering, centers, k, present_labels):
        assert k > 1

        n = data.shape[0] # num of samples
        d = data.shape[1] # dimension of data
        best_ratio = np.inf # M/min(C1, C2)
        coordinate, threshold = 0, np.NINF 
        u, clusters_count = np.unique(clustering, return_counts=True)
        num_of_labels = u.size # since the tree is build recursively, data separated from it's cluster will not have a cluster present.        
        mapping = dict(zip(u.tolist(), range(num_of_labels))) # since the tree is built recursively, a numbering 0,1,... is not guaranteed.
                                                              # clusters_count[i] = num of samples in cluster mapping[i]
        
        for i in range(d):
            data_ordering = data[:,i].argsort()
            centers_ordering = centers[:,i].argsort()
            curr_col = data[data_ordering, i]
            clustering_ordered = clustering[data_ordering]
            centers_ordered = centers[centers_ordering]
            indices_ordered = present_labels[centers_ordering]

            centers_threshold = -1 # final center (ordered by i'th coordinate) that is to the left of the cut.
            samples_location = np.zeros(num_of_labels) # samples_location[i] = number of samples that belongs to cluster mapping[i] in left side of cut.
            
            for j in range(n):
                samples_location[mapping[clustering_ordered[j]]] += 1 # x_j was moved to the left side of the cut.
                while centers_threshold < k-1 and centers_ordered[centers_threshold+1, i] <= curr_col[j]:
                    centers_threshold += 1
                
                if centers_threshold >= 0:
                    left_side_clusters = [mapping[c] for c in indices_ordered[:centers_threshold+1]] # This sucks
                    C1 = np.sum(samples_location[left_side_clusters]) 
                else:
                    continue

                if centers_threshold < k-1:
                    right_side_clusters = [mapping[c] for c in indices_ordered[centers_threshold+1:]] # This sucks
                    C2 = np.sum((clusters_count-samples_location)[right_side_clusters]) 
                else:
                    continue

                ratio = (n-C1-C2) / min(C1, C2)
                

                if ratio < best_ratio:
                    coordinate, threshold = i, curr_col[j]
                    best_ratio = ratio
        #print(best_ratio, coordinate, threshold)
        return best_ratio, coordinate, threshold
    
class IMM(ExplainableTree):
    def single_partition(self, data, clustering, centers, k, present_labels):
        assert k > 1

        n = data.shape[0] # num of samples
        d = data.shape[1] # dimension of data
        best_ratio = np.inf # mistakes
        coordinate, threshold = 0, np.NINF 
        u, clusters_count = np.unique(clustering, return_counts=True)
        num_of_labels = u.size # since the tree is build recursively, data separated from it's cluster will not have a cluster present.        
        mapping = dict(zip(u.tolist(), range(num_of_labels))) # since the tree is built recursively, a numbering 0,1,... is not guaranteed.
                                                              # clusters_count[i] = num of samples in cluster mapping[i]
        
        for i in range(d):
            data_ordering = data[:,i].argsort()
            centers_ordering = centers[:,i].argsort()
            curr_col = data[data_ordering, i]
            clustering_ordered = clustering[data_ordering]
            centers_ordered = centers[centers_ordering]
            indices_ordered = present_labels[centers_ordering]

            centers_threshold = -1 # final center (ordered by i'th coordinate) that is to the left of the cut.
            samples_location = np.zeros(num_of_labels) # samples_location[i] = number of samples that belongs to cluster mapping[i] in left side of cut.
            
            for j in range(n):
                samples_location[mapping[clustering_ordered[j]]] += 1 # x_j was moved to the left side of the cut.
                while centers_threshold < k-1 and centers_ordered[centers_threshold+1, i] <= curr_col[j]:
                    centers_threshold += 1
                
                if centers_threshold >= 0:
                    left_side_clusters = [mapping[c] for c in indices_ordered[:centers_threshold+1]] # This sucks
                    C1 = np.sum(samples_location[left_side_clusters]) 
                else:
                    continue

                if centers_threshold < k-1:
                    right_side_clusters = [mapping[c] for c in indices_ordered[centers_threshold+1:]] # This sucks
                    C2 = np.sum((clusters_count-samples_location)[right_side_clusters]) 
                else:
                    continue
                
                ratio = n-C1-C2
                
                if ratio < best_ratio:
                    coordinate, threshold = i, curr_col[j]
                    best_ratio = ratio
        #print(best_ratio, coordinate, threshold)
        return best_ratio, coordinate, threshold
    
class EMN(ExplainableTree):
    def single_partition(self, data, clustering, centers, k, present_labels):
        assert k > 1

        n = data.shape[0] # num of samples
        d = data.shape[1] # dimension of data
        best_ratio = np.inf # mistakes
        coordinate, threshold = 0, np.NINF 
        u, clusters_count = np.unique(clustering, return_counts=True)
        num_of_labels = u.size # since the tree is build recursively, data separated from it's cluster will not have a cluster present.        
        mapping = dict(zip(u.tolist(), range(num_of_labels))) # since the tree is built recursively, a numbering 0,1,... is not guaranteed.
                                                              # clusters_count[i] = num of samples in cluster mapping[i]
        
        for i in range(d):
            data_ordering = data[:,i].argsort()
            centers_ordering = centers[:,i].argsort()
            curr_col = data[data_ordering, i]
            clustering_ordered = clustering[data_ordering]
            centers_ordered = centers[centers_ordering]
            indices_ordered = present_labels[centers_ordering]

            centers_threshold = -1 # final center (ordered by i'th coordinate) that is to the left of the cut.
            samples_location = np.zeros(num_of_labels) # samples_location[i] = number of samples that belongs to cluster mapping[i] in left side of cut.
            
            for j in range(n):
                samples_location[mapping[clustering_ordered[j]]] += 1 # x_j was moved to the left side of the cut.
                while centers_threshold < k-1 and centers_ordered[centers_threshold+1, i] <= curr_col[j]:
                    centers_threshold += 1
                
                if centers_threshold >= 0:
                    left_side_clusters = [mapping[c] for c in indices_ordered[:centers_threshold+1]] # This sucks
                    C1 = np.sum(samples_location[left_side_clusters]) 
                else:
                    continue

                if centers_threshold < k-1:
                    right_side_clusters = [mapping[c] for c in indices_ordered[centers_threshold+1:]] # This sucks
                    C2 = np.sum((clusters_count-samples_location)[right_side_clusters]) 
                else:
                    continue
                
                ratio = (n-C1-C2) / min(len(left_side_clusters), len(right_side_clusters))
                
                if ratio < best_ratio:
                    coordinate, threshold = i, curr_col[j]
                    best_ratio = ratio
        #print(best_ratio, coordinate, threshold)
        return best_ratio, coordinate, threshold
    
