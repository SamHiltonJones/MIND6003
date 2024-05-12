import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import tf_transformations
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Quaternion
from std_msgs.msg import Header
import tf2_geometry_msgs
import open3d as o3d

class BallDetectionNode(Node):
    def __init__(self):
        super().__init__('ball_detection_node')
        self.get_logger().info("Initializing Ball Detection Node")
        self.subscription = self.create_subscription(Image, '/camera1/image_raw', self.image_callback, 10)
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.bridge = CvBridge()
        self.latest_image = None
        self.timer = self.create_timer(10.0, self.timer_callback)
        self.get_logger().info("Subscription and Timer set")
        self.ball_positions = {
            "ball_1": [1.00425, 0.948076],
            "ball_2": [0.966355, -2.98129],
            "ball_3": [-1.9949, -0.002344],
            "ball_4": [-1.01174, 2.99423],
            "ball_5": [3.03865, 0.845405],
            "ball_6": [-3.99068, -1.67433],
            "ball_7": [-2.98984, -3.99388],
            "ball_8": [-3.00162, 3.95973],
            "ball_9": [3.99293, 3.99831],
            "ball_10": [1.00303, 4.01006],
            "ball_11": [3.99293, -4.00216],
        }
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.image_width = 1200
        self.fov_horizontal = 1.8 
        self.fx, self.fy = self.calculate_focal_length(self.image_width, self.fov_horizontal)
        self.cx, self.cy = self.image_width / 2, self.image_width / 2

        self.depth_subscription = self.create_subscription(
            Image, 
            '/depth_camera/depth/image_raw',
            self.depth_image_callback, 
            10
        )
        self.get_logger().info("Subscribed to /depth_camera/depth/image_raw")
        self.depth_image = None
        self.timer = self.create_timer(10.0, self.process_depth_image)

    def depth_image_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f"Error in depth image conversion: {e}")

    def process_depth_image(self):
        if self.depth_image is not None:
            self.get_logger().info("Processing depth image")
            point_cloud = self.depth_to_point_cloud(self.depth_image)
            self.save_point_cloud(point_cloud, "output_point_cloud.ply")
        else:
            self.get_logger().warn("No depth image available to process")

    def depth_to_point_cloud(self, depth_image):
        height, width = depth_image.shape[:2]
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        x, y = np.meshgrid(x, y)
        z = depth_image
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy

        points = np.stack((x, y, z), axis=2).reshape(-1, 3)

        valid_points = points[z.ravel() > 0]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(valid_points)

        return pc

    def visualize_point_cloud(self, point_cloud):
        o3d.visualization.draw_geometries([point_cloud]) 

    def save_point_cloud(self, point_cloud, file_name):
        if point_cloud is not None:
            o3d.io.write_point_cloud(file_name, point_cloud)
            self.get_logger().info(f"Point cloud saved as: {file_name}")
        else:
            self.get_logger().warn("No point cloud to save.")

    def calculate_focal_length(self, image_width, fov_horizontal):
        fx = image_width / (2 * math.tan(fov_horizontal / 2))
        fy = fx
        return fx, fy

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def timer_callback(self):
        self.get_logger().info("Timer callback triggered")
        if self.latest_image is not None:
            self.process_image(self.latest_image)
        else:
            self.get_logger().warn("No image available to process")

    def process_image(self, image_msg):
        cv_image = self.latest_image
        if cv_image is None:
            self.get_logger().warn("No image available to process")
            return

        detected_ball_position, radius = self.detect_ball(cv_image) 
        if detected_ball_position and radius:
            cv2.circle(cv_image, tuple(detected_ball_position), radius, (0, 255, 0), 2)
            cv2.imwrite('detected_image.png', cv_image)

            estimated_distance = self.estimate_depth(detected_ball_position)

            self.get_logger().info(f"Detected ball at position: {detected_ball_position}, radius: {radius}, estimated distance: {estimated_distance:.2f}m")
            
            estimated_ball_id = self.estimate_ball_identity(detected_ball_position)
            if estimated_ball_id is not None:
                self.get_logger().info(f"Estimated ball ID: {estimated_ball_id}, estimated distance: {estimated_distance:.2f}m")
                robot_position = self.calculate_robot_position(cv_image, detected_ball_position, estimated_ball_id)
                if robot_position:
                    self.publish_robot_position(robot_position)
                    self.get_logger().info(f"Published robot position: {robot_position}, estimated distance to ball: {estimated_distance:.2f}m")
                else:
                    self.get_logger().info("Robot position could not be calculated.")
            else:
                self.get_logger().info("No valid ball ID estimated.")
        else:
            self.get_logger().info("No ball detected in image.")

    def detect_ball(self, image):
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = None
        largest_contour_area = 0

        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

                if (100 < area < 20000) and circularity > 0.8:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        radius = int(math.sqrt(area / math.pi))
                        return [cx, cy], radius
        return None, None


    def estimate_ball_identity(self, detected_position):
        estimated_depth = self.estimate_depth(detected_position)
        world_position = self.pixel_to_world(detected_position[0], detected_position[1], estimated_depth)

        if world_position is None:
            self.get_logger().warn("Could not determine world position.")
            return None

        closest_ball = None
        min_distance = float('inf')
        for ball_id, position in self.ball_positions.items():
            world_position_2d = np.array([world_position[0], world_position[1]])
            position_2d = np.array(position)
            distance = np.linalg.norm(position_2d - world_position_2d)
            if distance < min_distance:
                min_distance = distance
                closest_ball = ball_id

        if closest_ball:
            self.get_logger().info(f"Closest ball determined: {closest_ball}")
            return closest_ball
        return None


    def pixel_to_world(self, pixel_x, pixel_y, depth):
        normalized_x = (pixel_x - self.cx) / self.fx
        normalized_y = (pixel_y - self.cy) / self.fy
        camera_x = normalized_x * depth
        camera_y = normalized_y * depth
        camera_z = depth

        try:
            if self.tf_buffer.can_transform('odom', 'camera', rclpy.time.Time()):
                trans = self.tf_buffer.lookup_transform('odom', 'camera', rclpy.time.Time())
                
                translation = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
                rotation = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
                
                transform_matrix = tf_transformations.compose_matrix(translate=translation, angles=tf_transformations.euler_from_quaternion(rotation))

                camera_coords = np.array([camera_x, camera_y, camera_z, 1])
                world_coords = np.dot(transform_matrix, camera_coords)
                return world_coords[:3]
            else:
                self.get_logger().warn("Transformation from 'odom' to 'camera' not available")
                return None
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Error in transformation: {e}")
            return None


    def estimate_depth(self, ball_pixel_position):
        ball_radius_in_pixels = self.calculate_ball_radius_in_pixels(self.latest_image, ball_pixel_position)
        known_ball_diameter = 0.2  
        return (known_ball_diameter * self.fx) / (2 * ball_radius_in_pixels)

    def calculate_ball_radius_in_pixels(self, image, ball_pixel_position):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            closest_contour = min(contours, key=lambda c: cv2.pointPolygonTest(c, tuple(ball_pixel_position), True))
            x, y, w, h = cv2.boundingRect(closest_contour)
            return (w + h) / 4.0

    def calculate_robot_position(self, image, ball_pixel_position, ball_id):
        self.get_logger().info(f"Calculating robot position for ball ID {ball_id}")
        distance_to_ball = self.estimate_depth(ball_pixel_position)
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera', rclpy.time.Time())
            self.get_logger().info("Transformation between base_link and camera found")
            angle_to_ball = math.atan2(ball_pixel_position[1], ball_pixel_position[0])
            ball_world_position = self.ball_positions[ball_id]
            robot_world_x = ball_world_position[0] - distance_to_ball * math.cos(angle_to_ball + trans.transform.rotation.z)
            robot_world_y = ball_world_position[1] - distance_to_ball * math.sin(angle_to_ball + trans.transform.rotation.z)
            return [robot_world_x, robot_world_y, angle_to_ball]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().info("Could not calculate robot position")
            return None

    def publish_robot_position(self, position):
        self.get_logger().info(f"Publishing robot position: {position}")
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = position[0]
        pose_msg.pose.pose.position.y = position[1]
        pose_msg.pose.pose.position.z = 0.2
        orientation_quat = self.angle_to_quaternion(position[2])
        pose_msg.pose.pose.orientation.x = orientation_quat[0]
        pose_msg.pose.pose.orientation.y = orientation_quat[1]
        pose_msg.pose.pose.orientation.z = orientation_quat[2]
        pose_msg.pose.pose.orientation.w = orientation_quat[3]
        self.pose_publisher.publish(pose_msg)

    def angle_to_quaternion(self, yaw_angle):
        quat = tf_transformations.quaternion_from_euler(0, 0, yaw_angle)
        return quat

def main(args=None):
    rclpy.init(args=args)
    relocal = BallDetectionNode()
    rclpy.spin(relocal)
    relocal.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
