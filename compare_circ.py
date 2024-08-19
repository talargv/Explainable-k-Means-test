import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
import numpy as np

from general_graph import ExplainableGraphBased, get_nearest_neighbors
from algo import EMN
from play import plot_tree

def generate_dataset():
    factor = 0.2
    strip = 0.05
    n = 1000
    random_state = None
    k = 2

    data = make_circles(n_samples=n, random_state=random_state, factor= factor)[0]
    data = data[np.abs(data[:,1]) > strip]

    return data, k

def train_and_compare():
    data, k = generate_dataset()
    neighbors = 7

    g_tree = ExplainableGraphBased()
    nn_graph = get_nearest_neighbors(data, neighbors)
    g_tree.train(data, nn_graph, k)

    emn_tree = EMN()
    model = KMeans(n_clusters = k).fit(data)
    clustering_kmeans = model.labels_
    centers_kmeans = model.cluster_centers_
    emn_tree.train(data, clustering_kmeans, centers_kmeans)

    emn_clustering = emn_tree.clustering

    box = (np.min(data), np.max(data)) 

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,15))
    #plt.figure(0)
    colors = ['orange', 'blue']

    for cl,color in enumerate(colors):
        indices = emn_clustering == cl
        axes[0].scatter(data[indices, 0], data[indices, 1], marker='.', color=color, s=10)

    axes[1].scatter(data[:,0], data[:,1], marker='.', color='blue', s=10)

    axes[0].set_title("EMN")
    plot_tree(emn_tree.tree, box, box, axes[0])

    axes[1].set_title("Graph-Based")
    plot_tree(g_tree.tree, box, box, axes[1])

    plt.show()

train_and_compare()