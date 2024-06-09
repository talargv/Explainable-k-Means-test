from algo import *

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

def plot_cut(cut: Cut, x_box, y_box):
    if cut.left == None and cut.right == None:
        return
    
    assert np.isfinite(cut.threshold)

    if cut.coordinate == 0:
        plt.plot([cut.threshold, cut.threshold], y_box, linewidth=1, c="#990000")
        plot_cut(cut.right, (cut.threshold, x_box[1]), y_box)
        plot_cut(cut.left, (x_box[0], cut.threshold), y_box)
    else:
        plt.plot(x_box, [cut.threshold, cut.threshold], linewidth=1, c="#990000")
        plot_cut(cut.right, x_box, (cut.threshold, y_box[1]))
        plot_cut(cut.left, x_box, (y_box[0], cut.threshold))
    

def plot_tree(tree, x_box, y_box):
    plot_cut(tree.root, x_box, y_box)

k = 4
box = (-10.0, 10.0)

data, clustering, centers = make_blobs(n_samples=4000, centers=k, return_centers=True, center_box=box)

box = (np.min(data), np.max(data)) # data might be out of box.

plt.figure(0)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]

for cl,color in enumerate(colors):
    indices = clustering == cl
    plt.scatter(data[indices, 0], data[indices, 1], marker='.', c=color, s=10)

plt.scatter(centers[:, 0], centers[:, 1], c='b', s=50)

tree = Spectral()
tree.train(data, clustering, centers)
plot_tree(tree.tree, box, box)

plt.show()

