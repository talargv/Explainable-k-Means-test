import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from test_sets import *
from algo import *
from sklearn.cluster import KMeans

COLORS = list(mcolors.CSS4_COLORS.keys())

def test(data, k):
    assert (k-1)*5 < len(COLORS) # otherwise handle color choice differently
    assert data.shape[1] == 2 # needs some rewriting for 2\3 dimensions

    model = KMeans(n_clusters=k).fit(data)

    centers = model.cluster_centers_
    clustering = model.labels_

    best_ratio, coordinate, threshold = single_partition(data, clustering, centers, k)

    print(f"Best ratio - {best_ratio}")
    print(f"Coordinate - {coordinate}\nThreshold - {threshold}")

    for d in range(k):
        plt.plot(data[clustering == d,0], data[clustering == d,1], 'o', c=COLORS[5*d]) # WILL NOT WORK IN HIGH k
        plt.plot(centers[d,0], centers[d,1], '*', c=COLORS[5*d]) # WILL NOT WORK IN HIGH k
#plt.plot(data[clustering == 1,0], data[clustering == 1,1], 'ob')

    dummy = np.linspace(np.min(data,axis=1-coordinate),np.max(data,axis=1-coordinate),20)
    if coordinate:
        plt.plot(dummy, [threshold for _ in range(20)], c='r')
    else:
        plt.plot([threshold for _ in range(20)], dummy, c='r')

    plt.show()

test(zero_error_y_cut(), 3)


