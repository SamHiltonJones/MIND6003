import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

array1 = np.load('numpy_point_cloud.npy')
array2 = np.load('point_cloud_data.npy')

array1 = array1[~np.isinf(array1).any(axis=1)]
array2 = array2[~np.isinf(array2).any(axis=1)]

array2 = array2[array2[:, 2] < 6]

theta_z = -np.pi/2 
theta_y = 0  
theta_x = -np.pi/2 

Rx = np.array([
    [1, 0, 0],
    [0, np.cos(theta_x), -np.sin(theta_x)],
    [0, np.sin(theta_x), np.cos(theta_x)]
])

Ry = np.array([
    [np.cos(theta_y), 0, np.sin(theta_y)],
    [0, 1, 0],
    [-np.sin(theta_y), 0, np.cos(theta_y)]
])

Rz = np.array([
    [np.cos(theta_z), -np.sin(theta_z), 0],
    [np.sin(theta_z), np.cos(theta_z), 0],
    [0, 0, 1]
])

rotation_matrix_green = Rz @ Ry @ Rx

array2_rotated = array2 @ rotation_matrix_green.T

translation_vector = np.array([0.4, 0.0, 0.5]) 
array2_rotated += translation_vector

num_samples = int(0.01 * len(array2_rotated))
indices = np.random.choice(len(array2_rotated), num_samples, replace=False)
sampled_array2_rotated = array2_rotated[indices]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='red', marker='o', label='Array 1 (Original)')

ax.scatter(sampled_array2_rotated[:, 0], sampled_array2_rotated[:, 1], sampled_array2_rotated[:, 2], c='green', marker='o', label='Array 2 (Rotated and Translated)')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D Scatter Plot of Two Point Clouds - Adjusted')
plt.legend()
plt.show()



from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def find_differences(array1, array2, threshold=0.2):
    tree1 = cKDTree(array1)
    tree2 = cKDTree(array2)
    
    distances1, _ = tree2.query(array1)
    distances2, _ = tree1.query(array2)
    
    significant_diff1 = array1[distances1 > threshold]
    significant_diff2 = array2[distances2 > threshold]
    
    significant_diff_combined = np.vstack((significant_diff1, significant_diff2))
    
    np.save('significant_differences.npy', significant_diff_combined)
    
    return significant_diff1, significant_diff2

significant_diff1, significant_diff2 = find_differences(array1, sampled_array2_rotated)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='red', alpha=0.5, label='Array 1 (Original)')
# ax.scatter(sampled_array2_rotated[:, 0], sampled_array2_rotated[:, 1], sampled_array2_rotated[:, 2], c='green', alpha=0.5, label='Array 2 (Rotated and Translated)')
ax.scatter(significant_diff1[:, 0], significant_diff1[:, 1], significant_diff1[:, 2], c='blue', label='Differences from Array 1')
ax.scatter(significant_diff2[:, 0], significant_diff2[:, 1], significant_diff2[:, 2], c='yellow', label='Differences from Array 2')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('Differences between Two Point Clouds')
plt.legend()
plt.show()


from scipy.spatial import cKDTree
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def find_differences_and_clusters(array1, array2, threshold=0.2, min_samples=10, eps=0.5):
    tree1 = cKDTree(array1)
    tree2 = cKDTree(array2)
    
    distances1, _ = tree2.query(array1)
    distances2, _ = tree1.query(array2)
    
    significant_diff1 = array1[distances1 > threshold]
    significant_diff2 = array2[distances2 > threshold]
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(significant_diff1)
    labels = db.labels_
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    significant_clusters = unique_labels[counts > min_samples]
    
    significant_cluster_points = significant_diff1[np.isin(labels, significant_clusters)]

    return significant_diff1, significant_diff2, significant_cluster_points

significant_diff1, significant_diff2, significant_clusters = find_differences_and_clusters(array1, sampled_array2_rotated)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='red', alpha=0.5, label='Array 1 (Original)')
ax.scatter(sampled_array2_rotated[:, 0], sampled_array2_rotated[:, 1], sampled_array2_rotated[:, 2], c='green', alpha=0.5, label='Array 2 (Rotated and Translated)')
# ax.scatter(significant_clusters[:, 0], significant_clusters[:, 1], significant_clusters[:, 2], c='blue', s=50, label='Significant Structures in Array 1')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('Identified Structures in Point Cloud Differences')
plt.legend()
plt.show()
