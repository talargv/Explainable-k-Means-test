import matplotlib.pyplot as plt
import numpy as np

from general_graph import ExplainableGraphBased, get_nearest_neighbors
from play import plot_tree
from algo import Tree,Cut

def generate_dataset():
    k = 4
    inner_n = 1000
    outer_n = 2000

    rng = np.random.default_rng()
    inner_th = rng.uniform(0, 2*np.pi, inner_n)
    inner_radius = rng.uniform(0, 1, inner_n).reshape((inner_n,1))
    inner = np.stack((np.sin(inner_th), np.cos(inner_th)), axis=-1) * inner_radius
    inner = inner[np.abs(inner[:,1]) > 0.15]
    outer_th = rng.uniform(0, 2*np.pi, outer_n)
    outer_radius = rng.uniform(1.5, 2,outer_n).reshape((outer_n,1))
    outer = np.stack((np.sin(outer_th), np.cos(outer_th)), axis=-1) * outer_radius
    outer = outer[np.abs(outer[:,0]) > 0.7]

    data = np.concatenate((inner, outer), axis=0)

    return data, k

def generate_good_tree():
    tree = Tree()
    tree.root.coordinate, tree.root.threshold = 0, -1.0
    cut1 = Cut(0, 1.0)
    cut2 = Cut(1, 0.0)

    tree.root.left = Cut()
    tree.root.right = cut1

    cut1.right = Cut()
    cut1.left = cut2

    cut2.left, cut2.right = Cut(), Cut()

    return tree


def train_and_plot():
    data, k = generate_dataset()
    neighbors = 7

    g_tree = ExplainableGraphBased()
    nn_graph = get_nearest_neighbors(data, neighbors)
    g_tree.train(data, nn_graph, k)

    better_tree = generate_good_tree()

    box = (np.min(data), np.max(data)) 

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,15))

    axes[0].scatter(data[:,0], data[:,1], marker='.', color='blue', s=10)
    axes[1].scatter(data[:,0], data[:,1], marker='.', color='blue', s=10)

    axes[0].set_title("Better Tree")
    plot_tree(better_tree, box, box, axes[0])

    axes[1].set_title("Graph-Based")
    plot_tree(g_tree.tree, box, box, axes[1])

    plt.show()

train_and_plot()