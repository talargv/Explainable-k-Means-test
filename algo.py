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
    def single_partition(self, data, clustering, centers, k):
        """Decide a single axis-aligned threshold cut for data given clusters
        
        Input: data: np.ndarray - An array of size (number of samples) x (dimension of data)

        clustering: np.ndarray - An array of size (number of samples) that represents the clusters,
        i.e. clustering[j] = i iff x_j in cluster i

        k: int - number of clusters (k > 0)

        centers: np.ndarray - An array of size (k) x (dimension of data)
        
        Return: 
        best_ratio: float - min_{axis aligned threshold cuts S} phi(S)
        (phi is some function of the cut according to the metric the algo dictates)

        coordinate: int - coordinate of cut

        threshold: int - threshold of cut
        """
        pass

    def extend(self, data, clustering, centers, cut):
        k = centers.shape[0]
        if k == 1:
            return
        
        _, coordinate, threshold = self.single_partition(data, clustering, centers, k)
        cut.coordinate, cut.threshold = coordinate, threshold
        cut.left, cut.right = Cut(), Cut()

        l_data_partition = data[:,coordinate] <= threshold
        l_data = data[l_data_partition]
        l_clustering = clustering[l_data_partition]
        l_centers = centers[centers[:,coordinate] <= threshold]

        assert l_centers.shape[0] > 0

        self.extend(l_data, l_clustering, l_centers, cut.left)

        r_data_ordering = data[:,coordinate] > threshold
        r_data = data[r_data_ordering]
        r_clustering = clustering[r_data_ordering]
        r_centers = centers[centers[:,coordinate] > threshold]

        assert r_centers.shape[0] > 0

        self.extend(r_data, r_clustering, r_centers, cut.right)

    def train(self, data, clustering, centers):
        self.extend(data, clustering, centers, self.tree.root)

class Spectral(ExplainableTree):
    def single_partition(self, data, clustering, centers, k):
        assert k > 1

        n = data.shape[0] # num of samples
        d = data.shape[1] # dimension of data
        best_ratio = np.inf # M/min(C1, C2)
        coordinate, threshold = 0, np.NINF 
        u, clusters_count = np.unique(clustering, return_counts=True)        
        mapping = dict(zip(u.tolist(), range(k))) # since the tree is built recursively, we might have k clusters
                                                  # that aren't numbered as 0,...,k-1
                                                  # clusters_count[i] = num of samples in cluster mapping[i]
        print(u)
        print(n)
        print(k)
        print(mapping)
        center_indices = np.array([m for m in range(k)])
        
        for i in range(d):
            data_ordering = data[:,i].argsort()
            centers_ordering = centers[:,i].argsort()
            curr_col = data[data_ordering, i]
            clustering_ordered = clustering[data_ordering]
            centers_ordered = centers[centers_ordering]
            indices_ordered = center_indices[centers_ordering]

            centers_threshold = -1 # final center (ordered by i'th coordinate) that is to the left of the cut.
            samples_location = np.zeros(k) # samples_location[i] = number of samples that belongs to cluster mapping[i] in left side of cut.
            
            for j in range(n):
                samples_location[mapping[clustering_ordered[j]]] += 1 # x_j was moved to the left side of the cut.
                while centers_threshold < k-1 and centers_ordered[centers_threshold+1, i] <= curr_col[j]:
                    centers_threshold += 1
                
                if centers_threshold >= 0:
                    left_side_clusters = indices_ordered[:centers_threshold+1]
                    C1 = np.sum(samples_location[left_side_clusters])
                else:
                    continue

                if centers_threshold < k-1:
                    right_side_clusters = indices_ordered[centers_threshold+1:]
                    C2 = np.sum((clusters_count-samples_location)[right_side_clusters])
                else:
                    continue

                ratio = (n-C1-C2) / min(C1, C2)
                

                if ratio < best_ratio:
                    coordinate, threshold = i, curr_col[j]
                    best_ratio = ratio
        print(best_ratio, coordinate, threshold)
        return best_ratio, coordinate, threshold
    
class IMM(ExplainableTree):
    def single_partition(self, data, clustering, centers, k):
        assert k > 1

        n = data.shape[0] # num of samples
        d = data.shape[1] # dimension of data
        best_ratio = np.inf # Mistakes
        coordinate, threshold = 0, np.NINF 
        clusters_count = np.bincount(clustering) # clusters_count[i] = num of samples in cluster i
        
        center_indices = np.array([m for m in range(k)])
        
        for i in range(d):
            data_ordering = data[:,i].argsort()
            centers_ordering = centers[:,i].argsort()
            curr_col = data[data_ordering, i]
            clustering_ordered = clustering[data_ordering]
            centers_ordered = centers[centers_ordering]
            indices_ordered = center_indices[centers_ordering]

            centers_threshold = -1 # final center (ordered by i'th coordinate) that is to the left of the cut.
            samples_location = np.zeros(k) # samples_location[i] = number of samples that belongs to cluster i in left side of cut.
            
            for j in range(n):
                mistakes = 0

                samples_location[clustering_ordered[j]] += 1 # x_j was moved to the left side of the cut.
                while centers_threshold < k-1 and centers_ordered[centers_threshold+1, i] <= curr_col[j]:
                    centers_threshold += 1
                
                if centers_threshold >= 0:
                    left_side_clusters = indices_ordered[:centers_threshold+1]
                    mistakes += np.sum((clusters_count-samples_location)[left_side_clusters])
                else:
                    mistakes += np.sum(samples_location)

                if centers_threshold < k-1:
                    right_side_clusters = indices_ordered[centers_threshold+1:]
                    mistakes += np.sum(samples_location[right_side_clusters])
                else:
                    mistakes += np.sum(clusters_count-samples_location)

                ratio = mistakes
                
                if ratio < best_ratio:
                    coordinate, threshold = i, curr_col[j]
                    best_ratio = ratio

        return best_ratio, coordinate, threshold
    
class EMN(ExplainableTree):
    def single_partition(self, data, clustering, centers, k):
        assert k > 1

        n = data.shape[0] # num of samples
        d = data.shape[1] # dimension of data
        best_ratio = np.inf # Mistakes / min{clusters on the right, clusters on the left}
        coordinate, threshold = 0, np.NINF 
        clusters_count = np.bincount(clustering) # clusters_count[i] = num of samples in cluster i
        
        center_indices = np.array([m for m in range(k)])
        
        for i in range(d):
            data_ordering = data[:,i].argsort()
            centers_ordering = centers[:,i].argsort()
            curr_col = data[data_ordering, i]
            clustering_ordered = clustering[data_ordering]
            centers_ordered = centers[centers_ordering]
            indices_ordered = center_indices[centers_ordering]

            centers_threshold = -1 # final center (ordered by i'th coordinate) that is to the left of the cut.
            samples_location = np.zeros(k) # samples_location[i] = number of samples that belongs to cluster i in left side of cut.
            
            for j in range(n):
                mistakes = 0

                samples_location[clustering_ordered[j]] += 1 # x_j was moved to the left side of the cut.
                while centers_threshold < k-1 and centers_ordered[centers_threshold+1, i] <= curr_col[j]:
                    centers_threshold += 1
                
                if centers_threshold >= 0:
                    left_side_clusters = indices_ordered[:centers_threshold+1]
                    mistakes += np.sum((clusters_count-samples_location)[left_side_clusters])
                else:
                    mistakes += np.sum(samples_location)

                if centers_threshold < k-1:
                    right_side_clusters = indices_ordered[centers_threshold+1:]
                    mistakes += np.sum(samples_location[right_side_clusters])
                else:
                    mistakes += np.sum(clusters_count-samples_location)

                l_clusters_count = centers_threshold + 2
                r_clusters_count = k - l_clusters_count
                ratio = mistakes / min(l_clusters_count, r_clusters_count)
                
                if ratio < best_ratio:
                    coordinate, threshold = i, curr_col[j]
                    best_ratio = ratio

        return best_ratio, coordinate, threshold
    
