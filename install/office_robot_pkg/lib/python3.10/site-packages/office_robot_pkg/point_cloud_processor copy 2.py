import os
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import tf_transformations
from ament_index_python.packages import get_package_share_directory  # Import the correct function

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(10.0, self.timer_callback)
        
        # Correctly use the get_package_share_directory function
        package_dir = get_package_share_directory('office_robot_pkg')
        self.input_npy_path = os.path.join(package_dir, 'point_cloud', 'whole_map_point_cloud.npy')
        self.output_npy_dir = os.path.join(package_dir, 'point_cloud', 'filtered_points')
        os.makedirs(self.output_npy_dir, exist_ok=True)
        self.get_logger().info('PointCloudProcessor initialized')


    def timer_callback(self):
        try:
            map_to_base_link_trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.process_point_cloud(map_to_base_link_trans, 'filtered_points.npy')
        except Exception as e:
            self.get_logger().info(f'Could not transform point cloud: {str(e)}')

    def process_point_cloud(self, trans, output_file_name):
        self.get_logger().info('Processing point cloud')
        rotation_quaternion = [trans.transform.rotation.x, trans.transform.rotation.y, 
                               trans.transform.rotation.z, trans.transform.rotation.w]
        translation = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
        euler_angles = tf_transformations.euler_from_quaternion(rotation_quaternion)
        transformation_matrix = tf_transformations.compose_matrix(translate=translation, angles=euler_angles)
        points = np.load(self.input_npy_path)
        transformed_points = self.apply_transformation(points, transformation_matrix)
        camera_direction = transformation_matrix[:3, :3] @ np.array([1, 0, 0])
        filtered_points = self.filter_points(transformed_points, camera_direction)
        np.save(os.path.join(self.output_npy_dir, output_file_name), filtered_points)
        self.get_logger().info(f'Filtered point cloud saved to: {os.path.join(self.output_npy_dir, output_file_name)}')

    def apply_transformation(self, points, transformation_matrix):
        homogeneous_coords = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = (transformation_matrix @ homogeneous_coords.T).T[:, :3]
        return transformed_points

    def filter_points(self, points, camera_direction, camera_height=0.2, horizontal_fov=np.deg2rad(140), vertical_fov=np.deg2rad(103), min_distance=0.01, max_distance=10.0):
        filtered_points = []
        for point in points:
            x, y, z = point[0], point[1], point[2] - camera_height
            distance = np.sqrt(x**2 + y**2 + z**2)
            if min_distance <= distance <= max_distance:
                point_direction = np.array([x, y, z]) / distance
                cos_angle = np.dot(point_direction, camera_direction)
                horizontal_angle = np.arctan2(y, x)
                vertical_angle = np.arctan2(z, np.sqrt(x**2 + y**2))
                if abs(horizontal_angle) < horizontal_fov / 2 and abs(vertical_angle) < vertical_fov / 2:
                    filtered_points.append(point)
        return np.array(filtered_points)

def main(args=None):
    rclpy.init(args=args)
    processor = PointCloudProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
