from algo import *

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import sklearn.metrics as skm

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

def get_k_cost(data, clustering, k):
    k_cost = 0
    for i in range(k):
        k_cost += get_cost(data[clustering == i, :])
    return k_cost    

def get_weighted_center_dist(data, clustering, centers, k):
    weighted_center_distance = 0
    for i in range(k):
        cluster_i_size = data[clustering == i, :].shape[0]
        for j in range(k):
            cluster_j_size = data[clustering == j, :].shape[0]
            center_dist = np.linalg.norm(centers[i,:] - centers[j,:]) ** 2

            weighted_center_distance += cluster_i_size * cluster_j_size * center_dist
    return weighted_center_distance / data.shape[0]

def get_clusterability(data, clustering_true, centers, k):
    k_cost = get_k_cost(data, clustering_true, k)
    one_cost = get_cost(data)
    weighted_center_distance = get_weighted_center_dist(data, clustering_true, centers, k)

    return (k_cost / (one_cost + weighted_center_distance)) ** 0.5

def train_and_plot(data, clustering, centers):
    k = centers.shape[0]
    box = (np.min(data), np.max(data)) # data might be out of box.

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,15))
    #plt.figure(0)
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

def eval_model(data, clustering_true, centers, tree_type, show=False):
    k = centers.shape[0]
    tree = tree_type()
    tree.train(data, clustering_true, centers)
    clustering = tree.clustering

    if show:
        tree_name = tree_type.__name__
        print(f"Metrics for {tree_name}:")
        print(f"    Silhouette score: {skm.silhouette_score(data, clustering)}")
        print(f"    Price of Explainability: {get_k_cost(data, clustering, k) / get_k_cost(data, clustering_true, k)}")
    
        assert abs(get_k_cost(data, clustering, k) - tree.cost) < 0.003
    return tree.cost

def vanilla_kmeans_metrics(data, clustering_true, centers, k, show=False):
    silouette = skm.silhouette_score(data, clustering_true)
    clusterability = get_clusterability(data, clustering_true, centers, k)
    if show:
        print("Metric for Vanilla k-Means:")
        print(f"    Silhouette score: {silouette}")
        print(f"    Clusterability: {clusterability}")

    return silouette, clusterability

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

def experiment_k(n_runs):
    models = [IMM, Spectral, EMN]
    NUM_OF_MODELS = len(models)
    seed = None
    k_list = [4, 10, 25, 50]
    box = (-5, 5) # The range of the coordinates of the centers
    std = 1 # Variance
    n_samples = 4000
    dimension = 10

    s_avgs = np.zeros(len(k_list))
    c_avgs = np.zeros(len(k_list))
    prices_avgs = np.zeros((NUM_OF_MODELS, len(k_list)))
    
    for (l, k) in enumerate(k_list):
        s_scores = np.zeros(n_runs)
        c_values = np.zeros(n_runs)
        prices = np.zeros((NUM_OF_MODELS, n_runs))

        for i in range(n_runs):
            data, _, centers = make_blobs(n_samples=n_samples, centers=k, cluster_std=std,
                                                    random_state=seed, return_centers=True, n_features=dimension, center_box=box)
            model = KMeans(n_clusters = k, init = centers).fit(data)

            clustering_true = model.labels_
            centers = model.cluster_centers_

            s_scores[i], c_values[i] = vanilla_kmeans_metrics(data, clustering_true, centers, k)

            k_cost = get_k_cost(data, clustering_true, k)

            for (j, model) in enumerate(models):
                prices[j,i] = eval_model(data, clustering_true, centers, model) / k_cost

        s_avgs[l] = np.sum(s_scores) / n_runs
        c_avgs[l] = np.sum(c_values) / n_runs
        prices_avgs[:,l] = np.sum(prices, axis=1) / n_runs

    #_, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,15))
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,15))
    axes[0].plot(k_list, prices_avgs[0,:], 'o-', color="red", label="IMM") #IMM
    axes[0].plot(k_list, prices_avgs[1,:], 'o-', color="blue", label="Spectral") #Spectral
    axes[0].plot(k_list, prices_avgs[2,:], 'o-', color="green", label="EMN") #EMN
    axes[0].set_title("Ratio of tree cost to k-Cost")
    axes[0].legend()
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Ratio")
    axes[1].plot(k_list, s_avgs, 'o-')
    axes[1].set_title("Silhouette score of k-Means")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Sillhouette Score")
    #axes[2].plot(k_list, c_avgs)

    plt.show()

def experiment_std(n_runs):
    models = [IMM, Spectral, EMN]
    NUM_OF_MODELS = len(models)
    seed = None
    k = 10
    box = (-10, 10) # The range of the coordinates of the centers
    std_list = [20,15,10,5,1] # Variance
    n_samples = 4000
    dimension = 10

    s_avgs = np.zeros(len(std_list))
    c_avgs = np.zeros(len(std_list))
    prices_avgs = np.zeros((NUM_OF_MODELS, len(std_list)))
    
    for (l, std) in enumerate(std_list):
        s_scores = np.zeros(n_runs)
        c_values = np.zeros(n_runs)
        prices = np.zeros((NUM_OF_MODELS, n_runs))

        for i in range(n_runs):
            data, _, centers = make_blobs(n_samples=n_samples, centers=k, cluster_std=std,
                                                    random_state=seed, return_centers=True, n_features=dimension, center_box=box)
            model = KMeans(n_clusters = k, init = centers).fit(data)

            clustering_true = model.labels_
            centers = model.cluster_centers_

            s_scores[i], c_values[i] = vanilla_kmeans_metrics(data, clustering_true, centers, k)

            k_cost = get_k_cost(data, clustering_true, k)

            for (j, model) in enumerate(models):
                prices[j,i] = eval_model(data, clustering_true, centers, model) / k_cost

        s_avgs[l] = np.sum(s_scores) / n_runs
        c_avgs[l] = np.sum(c_values) / n_runs
        prices_avgs[:,l] = np.sum(prices, axis=1) / n_runs

    #_, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,15))
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,15))
    """axes[0].plot(std_list, prices_avgs[0,:], color="red") #IMM
    axes[0].plot(std_list, prices_avgs[1,:], color="blue") #Spectral
    axes[0].plot(std_list, prices_avgs[2,:], color="green") #EMN
    axes[1].plot(std_list, s_avgs)
    axes[2].plot(std_list, c_avgs)"""

    axes[0].plot(std_list, prices_avgs[0,:], 'o-', color="red", label="IMM") #IMM
    axes[0].plot(std_list, prices_avgs[1,:], 'o-', color="blue", label="Spectral") #Spectral
    axes[0].plot(std_list, prices_avgs[2,:], 'o-', color="green", label="EMN") #EMN
    axes[0].set_title("Ratio of tree cost to k-Cost")
    axes[0].legend()
    axes[0].set_xlabel("Variance")
    axes[0].set_ylabel("Ratio")
    axes[1].plot(std_list, s_avgs, 'o-')
    axes[1].set_title("Silhouette score of k-Means")
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Sillhouette Score")

    plt.show()

if __name__ == "__main__":
    seed = None
    k = 50
    box = (-5, 5) # The range of the coordinates of the centers
    std = 1 # Variance
    n_samples = 4000
    dimension = 2

    data, _, centers = make_blobs(n_samples=n_samples, centers=k, cluster_std=std,
                                                    random_state=seed, return_centers=True, n_features=dimension, center_box=box)
    model = KMeans(n_clusters = k, init = centers).fit(data)

    clustering = model.labels_
    centers = model.cluster_centers_

    train_and_plot(data, clustering, centers)