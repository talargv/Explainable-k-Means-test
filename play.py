from algo import *

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def plot_cut(cut: Cut, x_box, y_box, ax):
    if cut.left == None and cut.right == None:
        return
    
    assert np.isfinite(cut.threshold)

    if cut.coordinate == 0:
        ax.plot([cut.threshold, cut.threshold], y_box, linewidth=1, c="#990000")
        plot_cut(cut.right, (cut.threshold, x_box[1]), y_box, ax)
        plot_cut(cut.left, (x_box[0], cut.threshold), y_box, ax)
    else:
        ax.plot(x_box, [cut.threshold, cut.threshold], linewidth=1, c="#990000")
        plot_cut(cut.right, x_box, (cut.threshold, y_box[1]), ax)
        plot_cut(cut.left, x_box, (y_box[0], cut.threshold), ax)
    

def plot_tree(tree, x_box, y_box, ax):
    plot_cut(tree.root, x_box, y_box, ax)

def get_cost(data):
    median_of_data = np.sum(data, axis=0) / data.shape[0]
    return np.sum(np.linalg.norm(data - median_of_data, axis=1) ** 2)

def train_and_plot(data, clustering, centers):
    box = (np.min(data), np.max(data)) # data might be out of box.

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,15))
    plt.figure(0)
    colors = plt.cm.hsv(np.linspace(0,1,k))

    for i in range(3):
        for cl,color in enumerate(colors):
            indices = clustering == cl
            axes[i].scatter(data[indices, 0], data[indices, 1], marker='.', color=color, s=10)

        axes[i].scatter(centers[:, 0], centers[:, 1], c='midnightblue', s=50)

    tree = IMM()
    tree.train(data, clustering, centers)
    axes[0].scatter(tree.centers[:, 0], tree.centers[:, 1], c='black', s=50)
    print(f"IMM cost: {tree.cost}")
    axes[0].set_title("IMM")
    plot_tree(tree.tree, box, box, axes[0])

    tree = Spectral()
    tree.train(data, clustering, centers)
    axes[1].scatter(tree.centers[:, 0], tree.centers[:, 1], c='black', s=50)
    print(f"Spectral cost: {tree.cost}")
    axes[1].set_title("Spectral")
    plot_tree(tree.tree, box, box, axes[1])

    tree = EMN()
    tree.train(data, clustering, centers)
    axes[2].scatter(tree.centers[:, 0], tree.centers[:, 1], c='black', s=50)
    print(f"EMN cost: {tree.cost}")
    axes[2].set_title("EMN")
    plot_tree(tree.tree, box, box, axes[2])

    plt.show()

def show_cost(data, clustering, centers):
    tree = IMM()
    tree.train(data, clustering, centers)
    print(f"IMM cost: {tree.cost}")

    tree = Spectral()
    tree.train(data, clustering, centers)
    print(f"Spectral cost: {tree.cost}")

    tree = EMN()
    tree.train(data, clustering, centers)
    print(f"EMN cost: {tree.cost}")


seed = None
k = 30
box = (-20, 20)
std = 5
n_samples = 4000
dimension = 50

data, clustering, centers = make_blobs(n_samples=n_samples, centers=k, cluster_std=std,
                                        random_state=seed, return_centers=True, n_features=dimension, center_box=box)
model = KMeans(n_clusters = k, init = centers).fit(data)

clustering = model.labels_
centers = model.cluster_centers_

one_cost = get_cost(data)
k_cost = 0
for i in range(k):
    k_cost += get_cost(data[clustering == i, :])
weighted_center_distance = 0
for i in range(k):
    cluster_i_size = data[clustering == i, :].shape[0]
    for j in range(k):
        cluster_j_size = data[clustering == j, :].shape[0]
        center_dist = np.linalg.norm(centers[i,:] - centers[j,:]) ** 2

        weighted_center_distance += cluster_i_size * cluster_j_size * center_dist
weighted_center_distance /= data.shape[0]

print(f"k-Cost: {k_cost}")
print(f"1-Cost: {one_cost}")
print(f"Thingy: {weighted_center_distance}")
print(f"Clusterability: {k_cost / (one_cost + weighted_center_distance)}")


show_cost(data, clustering, centers)
