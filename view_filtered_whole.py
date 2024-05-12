import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

array1 = np.load('/home/sam/tut/MIND6003/install/office_robot_pkg/share/office_robot_pkg/point_cloud/filtered_pcds/filtered_point_cloud.npy')
array2 = np.load('whole_map_point_cloud.npy')

filtered_array1 = array1[~np.isin(array1, [np.inf, -np.inf]).any(axis=1)]
filtered_array2 = array2[~np.isin(array2, [np.inf, -np.inf]).any(axis=1)]

num_samples1 = int(0.5 * len(filtered_array1)) 
num_samples2 = int(0.1 * len(filtered_array2)) 

sampled_indices1 = np.random.choice(filtered_array1.shape[0], num_samples1, replace=False)
sampled_indices2 = np.random.choice(filtered_array2.shape[0], num_samples2, replace=False)

sampled_array1 = filtered_array1[sampled_indices1]
sampled_array2 = filtered_array2[sampled_indices2]

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for first array
ax.scatter(sampled_array1[:, 0], sampled_array1[:, 1], sampled_array1[:, 2], c='r', marker='o', label='Filtered Points')
# Scatter plot for second array
ax.scatter(sampled_array2[:, 0], sampled_array2[:, 1], sampled_array2[:, 2], c='b', marker='^', label='Whole Map Points')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('Comparison of Two Point Clouds')
plt.legend()
plt.show()
