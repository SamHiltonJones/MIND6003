import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import apriltag
import math
import tf_transformations

class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__('apriltag_detector_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('fov', 1.8),
                ('width', 1200),
                ('height', 1200),
                ('tag_world_position', [2.2, 1.2, 1.0]), 
                ('tag_world_orientation', [0.7071, 0.0, 0.7071, 0.0]) 
            ])

        fov = self.get_parameter('fov').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        self.fx = width / (2.0 * math.tan(fov / 2.0))
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.tag_world_position = self.get_parameter('tag_world_position').value
        self.tag_world_orientation = self.get_parameter('tag_world_orientation').value

        self.bridge = CvBridge()
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
        self.publisher = self.create_publisher(PoseStamped, '/robot_pose', 10)

        self.subscription = self.create_subscription(
            Image,
            '/camera1/image_raw',
            self.image_callback,
            10)

        self.get_logger().info('AprilTag Detector Node has started.')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {str(e)}')
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        for detection in detections:
            pose, _ = self.estimate_pose(detection)
            self.get_logger().info(f'Detected AprilTag ID: {detection.tag_id} at Pose: {pose}')
            self.publish_robot_pose(pose)

    def estimate_pose(self, detection):
        camera_matrix = np.array([[self.fx, 0, self.cx],
                                  [0, self.fy, self.cy],
                                  [0, 0, 1]], dtype=float)
        dist_coeffs = np.zeros((4, 1))
        tag_size = 1.0  
        obj_pts = np.array([
            [-tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, tag_size / 2, 0],
            [-tag_size / 2, tag_size / 2, 0]
        ])
        img_pts = np.array(detection.corners, dtype=float)

        _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
        return (rvec, tvec), detection.tag_id

    def publish_robot_pose(self, pose):

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world' 
        msg.pose.position.x = float(self.tag_world_position[0] - pose[1][0])
        msg.pose.position.y = float(self.tag_world_position[1] - pose[1][1])
        msg.pose.position.z = float(self.tag_world_position[2] - pose[1][2])
        msg.pose.orientation.x = float(self.tag_world_orientation[0])
        msg.pose.orientation.y = float(self.tag_world_orientation[1])
        msg.pose.orientation.z = float(self.tag_world_orientation[2])
        msg.pose.orientation.w = float(self.tag_world_orientation[3])
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
