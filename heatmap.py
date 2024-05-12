import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

array1 = np.load('/home/sam/tut/MIND6003/updated_whole_map_point_cloud.npy')
array2 = np.load('/home/sam/tut/MIND6003/whole_map_point_cloud.npy')

conditions = lambda arr: (~np.isinf(arr).any(axis=1) & ~np.isnan(arr).any(axis=1) &
                          ((arr[:, 0] > -6) & (arr[:, 0] < 5)) &
                          ((arr[:, 1] > -5) & (arr[:, 1] < 5)) & 
                          (arr[:, 2] > 0))

array1 = array1[conditions(array1)]
array2 = array2[conditions(array2)]

array1_xy = array1[:, :2]
array2_xy = array2[:, :2]

tree1_xy = cKDTree(array1_xy)
tree2_xy = cKDTree(array2_xy)

indices_to_keep = tree2_xy.query_ball_point(array1_xy, r=0.05)
indices_to_remove = set(range(len(array2_xy))) - set(np.concatenate(indices_to_keep))

removed_points = array2_xy[list(indices_to_remove)]

x_min, x_max, y_min, y_max = -6, 5, -5, 5
grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j] 

kde = gaussian_kde(array1_xy.T, bw_method='silverman')
grid_kde = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))

norm = plt.Normalize(grid_kde.min(), grid_kde.max())

plt.figure(figsize=(10, 8))
heatmap = plt.imshow(grid_kde.reshape(grid_x.shape).T, origin='lower', cmap='viridis', norm=norm, extent=(x_min, x_max, y_min, y_max))
plt.colorbar(heatmap, label='KDE of Remaining Points')

if removed_points.size > 0:
    removed_points_kde = gaussian_kde(removed_points.T, bw_method='silverman')
    removed_heatmap = removed_points_kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
    norm_removed = plt.Normalize(removed_heatmap.min(), removed_heatmap.max())
    plt.imshow(removed_heatmap.reshape(grid_x.shape).T, origin='lower', cmap='viridis', alpha=0.5, norm=norm_removed, extent=(x_min, x_max, y_min, y_max))
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_removed, cmap='viridis'), label='KDE of Removed Points')

plt.scatter(array2_xy[:, 0], array2_xy[:, 1], s=1, c='red', label='Original Point Cloud') 
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Point Cloud Differences with Structural Outlines')
plt.legend()
plt.axis('equal') 
plt.grid(True)
plt.show()
