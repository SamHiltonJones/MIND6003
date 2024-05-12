import os
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import numpy as np
import tf_transformations
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.timer_callback)

        package_dir = get_package_share_directory('office_robot_pkg')
        self.input_npz_path = os.path.join(package_dir, 'point_cloud', 'original_pcds', 'whole_map_point_cloud.npy')
        self.output_npz_dir = os.path.join(package_dir, 'point_cloud', 'filtered_pcds')
        os.makedirs(self.output_npz_dir, exist_ok=True)

        self.publisher = self.create_publisher(String, 'matrix_topic', 1)

        self.get_logger().info('PointCloudProcessor initialized')
        
    def timer_callback(self):
        try:
            map_to_base_link_trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.process_point_cloud(map_to_base_link_trans, 'filtered_point_cloud.npy')
        except Exception as e:
            self.get_logger().info('Could not transform point cloud: ' + str(e))

    def process_point_cloud(self, trans, output_file_name):
        # self.get_logger().info('Processing point cloud')

        rotation_quaternion = [trans.transform.rotation.x, trans.transform.rotation.y, 
                            trans.transform.rotation.z, trans.transform.rotation.w]
        translation = [-trans.transform.translation.x, trans.transform.translation.y, 
                    trans.transform.translation.z]

        euler_angles = tf_transformations.euler_from_quaternion(rotation_quaternion)
        euler_angles = (euler_angles[0], euler_angles[1], -euler_angles[2])
        transformation_matrix = tf_transformations.compose_matrix(translate=translation, 
                                                                angles=euler_angles)

        camera_offset_translation = [0.15, 0, 0.2]
        camera_transformation_matrix = tf_transformations.compose_matrix(translate=camera_offset_translation)
        
        final_transformation_matrix = np.dot(transformation_matrix, camera_transformation_matrix)

        points = np.load(self.input_npz_path)
        transformed_points = np.dot(points, final_transformation_matrix[:3, :3].T) + final_transformation_matrix[:3, 3]

        camera_direction = final_transformation_matrix[:3, :3] @ np.array([1, 0, 0])
        filtered_points = [point for point in transformed_points
                        if self.is_point_in_cone(point, camera_offset_translation[2], np.deg2rad(103), np.deg2rad(103), 
                                                    0.01, 10.0, camera_direction)]

        inverse_final_transformation_matrix = np.linalg.inv(final_transformation_matrix)

        matrix_str = np.array2string(inverse_final_transformation_matrix, separator=', ')
        matrix_msg = String(data=matrix_str)
        self.publisher.publish(matrix_msg)
        # self.get_logger().info('Inverse transformation matrix published')

        reverted_points = np.dot(filtered_points - final_transformation_matrix[:3, 3], inverse_final_transformation_matrix[:3, :3].T)

        output_npz_path = os.path.join(self.output_npz_dir, output_file_name)
        np.save(output_npz_path, np.array(reverted_points))
        # self.get_logger().info(f'Reverted point cloud saved to: {output_npz_path}')



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
