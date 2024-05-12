import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def optimal_clusters(data, min_k=2, max_k=8):
    if data.size == 0:
        return None, 0  

    scores = []
    inertias = []
    range_k = range(min_k, max_k + 1)

    for k in range_k:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=100, random_state=0).fit(data)
        inertias.append(kmeans.inertia_)
        if k > 1:
            scores.append(silhouette_score(data, kmeans.labels_))
        else:
            scores.append(-1) 

    if max_k == 1:
        optimal_k = 1
    else:
        optimal_k = np.argmax(scores) + min_k

    final_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=100, random_state=0).fit(data)
    return final_kmeans, optimal_k
def downsample_data(data, fraction=0.1):
    indices = np.random.choice(data.shape[0], int(data.shape[0] * fraction), replace=False)
    return data[indices]


array1 = np.load('/home/sam/tut/MIND6003/updated_whole_map_point_cloud.npy')
array2 = np.load('/home/sam/tut/MIND6003/whole_map_point_cloud.npy')

conditions = lambda arr: (~np.isinf(arr).any(axis=1) & ~np.isnan(arr).any(axis=1) &
                          ((arr[:, 0] > -6) & (arr[:, 0] < 5)) &
                          ((arr[:, 1] > -5) & (arr[:, 1] < 5)) & 
                          (arr[:, 2] > 0))
array1 = array1[conditions(array1)]
array2 = array2[conditions(array2)]
fraction = 0.3
array1 = downsample_data(array1, fraction)
array2 = downsample_data(array2, fraction)
tree1 = cKDTree(array1)
tree2 = cKDTree(array2)

indices_to_keep_from_array2 = tree2.query_ball_point(array1, r=0.05)
indices_removed = set(range(len(array2))) - set(np.concatenate(indices_to_keep_from_array2))

indices_to_keep_from_array1 = tree1.query_ball_point(array2, r=0.05)
indices_added = set(range(len(array1))) - set(np.concatenate(indices_to_keep_from_array1))

added_points = array1[list(indices_added)]
removed_points = array2[list(indices_removed)]

kmeans_added, k_added = optimal_clusters(added_points)
kmeans_removed, k_removed = optimal_clusters(removed_points, min_k = 1, max_k = 1)

distance_threshold = 0.5 
cluster_size_threshold = 10  

if k_added > 0:
    for idx, center in enumerate(kmeans_added.cluster_centers_):
        nearest_dist = tree2.query(center)[0]
        if nearest_dist > distance_threshold or np.sum(kmeans_added.labels_ == idx) >= cluster_size_threshold:
            print(np.sum(kmeans_added.labels_ == idx))
            if np.sum(kmeans_added.labels_ == idx) > 10000:
                print(f"Table added at {center[:2]}")
            else:
                print(f"Box added at {center[:2]}")

if k_removed > 0:
    for idx, center in enumerate(kmeans_removed.cluster_centers_):
        nearest_dist = tree1.query(center)[0]
        if nearest_dist > distance_threshold or np.sum(kmeans_removed.labels_ == idx) >= cluster_size_threshold:
            print(np.sum(kmeans_removed.labels_ == idx))
            if np.sum(kmeans_removed.labels_ == idx) > 10000:
                print(f"Table removed at {center[:2]}")
            else:
                print(f"Box removed at {center[:2]}")
print('done')
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.get_cmap('viridis', k_added)
removed_colors = plt.cm.get_cmap('inferno', k_removed)

if k_added > 0:
    for i in range(k_added):
        points = added_points[kmeans_added.labels_ == i]
        ax.scatter(points[:, 0], points[:, 1], color=colors(i), label=f'Object_added_{i+1}')

if k_removed > 0:
    for i in range(k_removed):
        points = removed_points[kmeans_removed.labels_ == i]
        ax.scatter(points[:, 0], points[:, 1], color=removed_colors(i), label=f'Object_removed_{i+1}')


ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('2D Map of Point Cloud with Cluster Labels')
ax.legend()
plt.show()
