import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct
import ctypes
import tf2_ros
from geometry_msgs.msg import PoseStamped, PointStamped
import tf2_geometry_msgs
import tf_transformations
from rclpy.timer import Rate
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
import sensor_msgs_py.point_cloud2 as pc2_py


class BallDetectionNode(Node):
    def __init__(self):
        super().__init__('ball_detection_node')
        self.pc_subscription = self.create_subscription(PointCloud2, '/depth_camera/points', self.pc_callback, 10)        
        self.ball_radius = 0.1
        self.pose_publisher = self.create_publisher(PoseStamped, '/estimated_robot_pose', 5)
        self.get_logger().info("Ball Detection Node initialized")
        self.ball_positions = {
            "ball_1": [1.00425, 0.948076, 0.05],
            "ball_2": [0.966355, -2.98129, 0.05],
            "ball_3": [-1.9949, -0.002344, 0.05],
            "ball_4": [-1.01174, 2.99423, 0.05],
            "ball_5": [3.03865, 0.845405, 0.05],
            "ball_6": [-3.99068, -1.67433, 0.05],
            "ball_7": [-2.98984, -3.99388, 0.05],
            "ball_8": [-3.00162, 3.95973, 0.05],
            "ball_9": [3.99293, 3.99831, 0.05],
            "ball_10": [1.00303, 4.01006, 0.05],
            "ball_11": [3.99293, -4.00216, 0.05],
        }
        self.tf_buffer = tf2_ros.Buffer()
        self.camera_horizontal_fov = 1.8
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.last_point_cloud = None
    
    def pc_callback(self, pc_msg):
        self.get_logger().info("Point cloud data received")
        gen = pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)

        xyz, rgb = [], []
        for x, y, z, rgb_float in gen:
            color = self.float_rgb_to_int(rgb_float)
            xyz.append([x, y, z])
            rgb.append(color / 255.0)

        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(xyz)
        out_pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud("cloud.ply", out_pcd)
        self.get_logger().info("Saved point cloud data to cloud.ply")
        self.get_logger().info("Point cloud data received")
        point_cloud = pc2_py.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        yellow_points = self.filter_yellow_points(point_cloud)

        if yellow_points:
            ball_position = self.find_ball_centroid(yellow_points)
            if ball_position is not None:
                distance = np.linalg.norm(ball_position)
                world_position = self.transform_point_to_world(ball_position)
                if world_position is None:
                    self.get_logger().info("World position could not be determined.")
                    return

                try:
                    if self.tf_buffer.can_transform('base_link', 'camera', rclpy.time.Time()):
                        trans = self.tf_buffer.lookup_transform('base_link', 'camera', rclpy.time.Time())
                        camera_orientation = [trans.transform.rotation.x, trans.transform.rotation.y, 
                                            trans.transform.rotation.z, trans.transform.rotation.w]
                        estimated_ball_id = self.estimate_ball_identity(world_position, camera_orientation)
                        if estimated_ball_id:
                            self.publish_robot_position(world_position, estimated_ball_id, distance)
                        else:
                            self.get_logger().info("Ball identity could not be determined.")
                    else:
                        self.get_logger().warn("Transformation from 'camera' to 'base_link' not available.")
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    self.get_logger().error(f"Error in obtaining camera orientation: {e}")

            else:
                self.get_logger().info("Yellow ball not detected.")
        else:
            self.get_logger().info("No yellow points found in point cloud.")


    def filter_yellow_points(self, point_cloud):
        yellow_lower = np.array([20, 100, 100], dtype=np.float32)
        yellow_upper = np.array([30, 255, 255], dtype=np.float32)
        yellow_points = []
        for point in point_cloud:
            x, y, z, rgb = point
            color = self.float_rgb_to_int(rgb)
            if np.all((yellow_lower <= color) & (color <= yellow_upper)):
                yellow_points.append([x, y, z])
        return yellow_points

    def find_ball_centroid(self, points):
        if points:
            centroid = np.mean(points, axis=0)
            return centroid
        return None

    def transform_point_to_world(self, point):
        try:
            if self.tf_buffer.can_transform('odom', 'camera', rclpy.time.Time()):
                trans = self.tf_buffer.lookup_transform('odom', 'camera', rclpy.time.Time())
                point_stamped = PointStamped()
                point_stamped.header.frame_id = "camera"
                point_stamped.header.stamp = rclpy.time.Time().to_msg()
                point_stamped.point.x = float(point[0])
                point_stamped.point.y = float(point[1])
                point_stamped.point.z = float(point[2])

                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, trans)
                return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]
            else:
                self.get_logger().warn("Transformation from 'camera' to 'odom' not available")
                return None
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Error in transformation: {e}")
            return None


    def is_in_camera_fov(self, ball_position, camera_orientation):
        euler_angles = tf_transformations.euler_from_quaternion(camera_orientation)
        forward_direction = [np.cos(euler_angles[1]) * np.cos(euler_angles[2]), 
                            np.sin(euler_angles[1]) * np.cos(euler_angles[2]), 
                            np.sin(euler_angles[2])]
        
        camera_world_position = self.tf_buffer.lookup_transform('odom', 'camera', rclpy.time.Time()).transform.translation
        camera_world_position = [camera_world_position.x, camera_world_position.y, camera_world_position.z]

        vector_to_ball = np.array(ball_position) - np.array(camera_world_position)
        vector_to_ball_normalized = vector_to_ball / np.linalg.norm(vector_to_ball)

        dot_product = np.dot(forward_direction, vector_to_ball_normalized)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        return angle <= self.camera_horizontal_fov / 2


    

    def estimate_ball_identity(self, world_position, camera_orientation):
        closest_ball = None
        min_distance = float('inf')

        for ball_id, position in self.ball_positions.items():
            if self.is_in_camera_fov(position, camera_orientation):
                distance = np.linalg.norm(np.array(world_position) - np.array(position))
                if distance < min_distance:
                    min_distance = distance
                    closest_ball = ball_id

        return closest_ball


    def publish_robot_position(self, world_position, ball_id, distance):
        robot_position = self.calculate_robot_position(world_position, ball_id, distance)
        if robot_position is not None and len(robot_position) > 0:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'odom'
            pose_msg.pose.position.x = robot_position[0]
            pose_msg.pose.position.y = robot_position[1]
            pose_msg.pose.position.z = 0.05
            self.pose_publisher.publish(pose_msg)
            self.get_logger().info(f"Published estimated robot position based on {ball_id}: {robot_position}")


    def calculate_robot_position(self, world_position, ball_id, distance):
        ball_world_position = self.ball_positions[ball_id]
        direction_vector = np.array(world_position) - np.array(ball_world_position)
        direction_vector /= np.linalg.norm(direction_vector)
        robot_position = np.array(ball_world_position) + direction_vector * distance
        return robot_position


    def float_rgb_to_int(self, float_rgb):
        s = struct.pack('>f', float_rgb)
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)
        return np.array([r, g, b], dtype=np.int32)

def main(args=None):
    rclpy.init(args=args)
    relocal = BallDetectionNode()
    rclpy.spin(relocal)
    relocal.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
