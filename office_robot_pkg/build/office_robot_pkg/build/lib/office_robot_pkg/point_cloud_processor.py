import os
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import open3d as o3d
import numpy as np
import tf_transformations
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
import ctypes
import struct

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(10.0, self.timer_callback)

        package_dir = get_package_share_directory('office_robot_pkg')
        self.input_pcd_path = os.path.join(package_dir, 'point_cloud', 'original_pcds', 'map.pcd')
        self.output_pcd_dir = os.path.join(package_dir, 'point_cloud', 'filtered_pcds')
        os.makedirs(self.output_pcd_dir, exist_ok=True)
        self.get_logger().info('PointCloudProcessor initialized')

  
    def timer_callback(self):
        try:
            map_to_base_link_trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            odom_to_base_link_trans = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())

            self.process_point_cloud(map_to_base_link_trans, 'map_view_map.pcd')
            self.process_point_cloud(odom_to_base_link_trans, 'odom_view_map.pcd')
        except Exception as e:
            self.get_logger().info('Could not transform point cloud: ' + str(e))


    def process_point_cloud(self, trans, output_file_name):
        self.get_logger().info('Processing point cloud')

        rotation_quaternion = [trans.transform.rotation.x, trans.transform.rotation.y, 
                            trans.transform.rotation.z, trans.transform.rotation.w]
        translation = [-trans.transform.translation.x, trans.transform.translation.y, 
                        trans.transform.translation.z]

        euler_angles = tf_transformations.euler_from_quaternion(rotation_quaternion)
        euler_angles = (euler_angles[0], euler_angles[1], -euler_angles[2])
        inverted_rotation_quaternion = tf_transformations.quaternion_from_euler(*euler_angles)

        transformation_matrix = tf_transformations.compose_matrix(translate=translation, 
                                                                angles=euler_angles)

        camera_offset_translation = [0.15, 0, 0.2]
        camera_transformation_matrix = tf_transformations.compose_matrix(translate=camera_offset_translation)
        
        final_transformation_matrix = np.dot(transformation_matrix, camera_transformation_matrix)

        self.get_logger().info(f'Position (base_link): {translation}')
        self.get_logger().info(f'Rotation (Inverted Quaternion, base_link): {inverted_rotation_quaternion}')

        self.get_logger().info(f'Trying to read point cloud from: {self.input_pcd_path}')
        pcd = o3d.io.read_point_cloud(self.input_pcd_path)

        if pcd.is_empty():
            self.get_logger().warning('Input point cloud is empty')
            return

        pcd.transform(final_transformation_matrix)

        camera_direction = final_transformation_matrix[:3, :3] @ np.array([1, 0, 0])

        filtered_points = [point for point in np.asarray(pcd.points) 
                        if self.is_point_in_cone(point, camera_offset_translation[2], np.deg2rad(87), np.deg2rad(58), 
                                                    0.28, 3.0, camera_direction)]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        self.get_logger().info(f'Filtered point cloud size: {len(filtered_points)}')

        # Adjust output file name to use .ply extension
        output_ply_file_name = output_file_name.replace('.pcd', '.ply')
        output_ply_path = os.path.join(self.output_pcd_dir, output_ply_file_name)

        # Write output as PLY
        o3d.io.write_point_cloud(output_ply_path, filtered_pcd, write_ascii=True)
        self.get_logger().info(f'Filtered point cloud written to: {output_ply_path}')


        output_pcd_path = os.path.join(self.output_pcd_dir, output_file_name)
        o3d.io.write_point_cloud(output_pcd_path, filtered_pcd)
        self.get_logger().info(f'Filtered point cloud written to: {output_pcd_path}')



    def rotate_about_z(self, rotation_matrix, angle):
        rot_z = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        return rot_z @ rotation_matrix

    def is_point_in_cone(self, point, camera_height, horizontal_fov, vertical_fov, min_distance, max_distance, camera_direction):
        x, y, z = point[0], point[1], point[2] - camera_height
        distance = np.sqrt(x**2 + y**2 + z**2)
        if min_distance <= distance <= max_distance:
            point_direction = np.array([x, y, z]) / distance
            cos_angle = np.dot(point_direction, camera_direction)
            horizontal_angle = np.arctan2(y, x)
            vertical_angle = np.arctan2(z, np.sqrt(x**2 + y**2))
            return abs(horizontal_angle) < horizontal_fov / 2 and abs(vertical_angle) < vertical_fov / 2
        return False

def main(args=None):
    rclpy.init(args=args)
    point_cloud_processor = PointCloudProcessor()
    rclpy.spin(point_cloud_processor)
    point_cloud_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
