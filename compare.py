import open3d as o3d
import numpy as np

def load_numpy_to_pcd(array):
    pcd = o3d.geometry.PointCloud()
    plain_array = np.vstack((array['x'], array['y'], array['z'])).T
    pcd.points = o3d.utility.Vector3dVector(plain_array)
    print("Point Cloud Loaded:", pcd)
    print("Number of points:", len(pcd.points))
    return pcd


def register_point_clouds(source_pcd, target_pcd, threshold=0.05):
    """Aligns two point clouds using the ICP algorithm."""
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return reg_p2p

def visualize_point_clouds(source, target):
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    vis.run()
    vis.destroy_window()


def load_and_compare_point_clouds(source_file, target_file):
    source_array = np.load(source_file)
    target_array = np.load(target_file)
    source_pcd = load_numpy_to_pcd(source_array)
    target_pcd = load_numpy_to_pcd(target_array)
    reg_p2p = register_point_clouds(source_pcd, target_pcd)
    print("Transformation matrix to align source to target:")
    print(reg_p2p.transformation)
    source_pcd.transform(reg_p2p.transformation)
    visualize_point_clouds(source_pcd, target_pcd)
    distances = source_pcd.compute_point_cloud_distance(target_pcd)
    average_distance = np.mean(distances)
    print("Average distance between point clouds:", average_distance)

source_file_path = '/home/sam/tut/MIND6003/structured_point_cloud.npy'
target_file_path = '/home/sam/tut/MIND6003/saved_point_cloud.npy'
load_and_compare_point_clouds(source_file_path, target_file_path)
