import numpy as np
import open3d as o3d

def create_point_cloud_from_file(file_path):
    try:
        # Read depth data from file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Parse depth data from comma-separated values
        depth_data = np.array([[int(val) for val in line.strip().split(',')] for line in lines])

        # Assuming depth values are in meters, create point cloud
        height, width = depth_data.shape
        fx = fy = 1.0  # Assume focal length of 1 for simplicity
        cx = width / 2
        cy = height / 2

        # Generate point cloud data from depth image
        rows, cols = np.indices(depth_data.shape)
        points = np.zeros((depth_data.size, 3))
        points[:, 0] = (cols - cx) * depth_data.ravel() / fx
        points[:, 1] = (rows - cy) * depth_data.ravel() / fy
        points[:, 2] = depth_data.ravel()

        # Convert numpy array to Open3D point cloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        return cloud
    except Exception as e:
        print("Error occurred while creating point cloud:", e)
        return None

def main():
    # File path to the depth data file
    file_path = 'myfile.txt'

    # Create point cloud from file
    point_cloud = create_point_cloud_from_file(file_path)

    if point_cloud is not None:
        # Save point cloud to PLY file
        o3d.io.write_point_cloud('point_cloud.ply', point_cloud, write_ascii=True)
        print("Point cloud saved successfully!")

if __name__ == '__main__':
    main()
