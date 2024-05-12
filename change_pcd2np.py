import open3d as o3d
import numpy as np

def load_pcd_to_numpy(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    return points

pcd_filepath = 'install/office_robot_pkg/share/office_robot_pkg/point_cloud/filtered_pcds/odom_view_map.pcd'
# pcd_filepath = 'install/office_robot_pkg/share/office_robot_pkg/point_cloud/original_pcds/map.pcd'
numpy_array = load_pcd_to_numpy(pcd_filepath)
print("Loaded point cloud as NumPy array with shape:", numpy_array.shape)

np.save('numpy_point_cloud', numpy_array)

print("First 10 elements of the NumPy array:\n", numpy_array[:10])

print("Statistics of the NumPy array:")
print("  Min:", np.min(numpy_array, axis=0))
print("  Max:", np.max(numpy_array, axis=0))
print("  Mean:", np.mean(numpy_array, axis=0))
print("  Std Deviation:", np.std(numpy_array, axis=0))
