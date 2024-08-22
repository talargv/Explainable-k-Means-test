from general_graph import *
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import sklearn.metrics as skm
from play import plot_tree

def train_and_plot(data, n_neighbors, k):
    box = (np.min(data), np.max(data)) # data might be out of box.

    fig, ax = plt.subplots(figsize=(15,15))
    #plt.figure(0)
    colors = plt.cm.hsv(np.linspace(0,1,k))

    tree = ExplainableGraphBased()
    graph = get_nearest_neighbors(data, n_neighbors)
    tree.train(data,graph,k)
    #tree.tree.print_structure()
    clustering = tree.predict(data)
    #print(tree.tree.root.cluster, tree.tree.root.coordinate, tree.tree.root.threshold)
    print(clustering)
    #print(f"Silhouette score: {skm.silhouette_score(data, clustering)}")
    

    plot_tree(tree.tree, box, box, ax)
    for cl,color in enumerate(colors):
        indices = clustering == cl + 1
        ax.scatter(data[indices, 0], data[indices, 1], marker='.', color=color, s=10)

    plt.show()
    
def generate_data(k):
    seed = None
    box = (-15, 15) # The range of the coordinates of the centers
    std = 1 # Variance
    n_samples = 3000
    dimension = 2

    data, _ = make_blobs(n_samples=n_samples, centers=k, cluster_std=std,
                                                    random_state=seed, n_features=dimension, center_box=box)
    
    return data

data = generate_data(50)
train_and_plot(data, 10, 50)