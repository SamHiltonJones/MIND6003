import cv2
import numpy as np
import math
import apriltag
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformBroadcaster

class AprilTagDetectionNode(Node):
    def __init__(self):
        super().__init__('april_tag_detection_node')
        # Set the tag size here (e.g., width of the tag in meters)
        self.declare_parameter('tag_size', 1.0)  # Adjust according to your actual tag size in simulation
        self.image_subscriber = self.create_subscription(Image, '/camera1/image_raw', self.image_callback, 10)
        self.pose_publisher = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.bridge = CvBridge()
        self.br = TransformBroadcaster(self)
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
        fov = 1.8  # Camera field of view in radians
        width, height = 1200, 1200  # Camera resolution
        fx = width / (2 * math.tan(fov / 2))
        fy = height / (2 * math.tan(fov / 2))
        cx, cy = width / 2, height / 2
        self.camera_params = (fx, fy, cx, cy)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detect(gray)
            if not detections:
                self.get_logger().info("No tags detected.")
                return

            for detection in detections:
                self.process_detection(detection, msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def process_detection(self, detection, stamp):
        try:
            pose, e0, e1 = self.detector.detection_pose(detection, self.camera_params, self.get_parameter('tag_size').value)
            if pose is None or pose.size == 0:
                self.get_logger().info("No valid pose could be computed.")
                return

            self.get_logger().info(f"Processing pose for tag {detection.tag_id} with shape {pose.shape}, pose: {pose}")
            rmat = pose[:3, :3]
            tvec = pose[:3, 3]
            quaternion = self.rotation_matrix_to_quaternion(rmat)
            self.publish_pose(stamp, tvec, quaternion)

        except Exception as e:
            self.get_logger().error(f"Error processing pose for tag {detection.tag_id}: {e}")
            self.get_logger().info(f"Pose data at error: {pose}")

    def publish_pose(self, stamp, tvec, quaternion):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "camera"
        pose_msg.pose.position.x = tvec[0]
        pose_msg.pose.position.y = tvec[1]
        pose_msg.pose.position.z = tvec[2]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        self.pose_publisher.publish(pose_msg)

    def rotation_matrix_to_quaternion(self, rmat):
        # Standard conversion algorithm to convert a rotation matrix to a quaternion
        q = np.zeros(4)
        trace = np.trace(rmat)
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (rmat[2, 1] - rmat[1, 2]) * s
            q[1] = (rmat[0, 2] - rmat[2, 0]) * s
            q[2] = (rmat[1, 0] - rmat[0, 1]) * s
        else:
            # Handling when trace is less than or equal to zero
            max_diag = np.argmax([rmat[0, 0], rmat[1, 1], rmat[2, 2]])
            if max_diag == 0:
                s = 2.0 * math.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2])
                q[3] = (rmat[2, 1] - rmat[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (rmat[0, 1] + rmat[1, 0]) / s
                q[2] = (rmat[0, 2] + rmat[2, 0]) / s
            elif max_diag == 1:
                s = 2.0 * math.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2])
                q[3] = (rmat[0, 2] - rmat[2, 0]) / s
                q[0] = (rmat[0, 1] + rmat[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (rmat[1, 2] + rmat[2, 1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1])
                q[3] = (rmat[1, 0] - rmat[0, 1]) / s
                q[0] = (rmat[0, 2] + rmat[2, 0]) / s
                q[1] = (rmat[1, 2] + rmat[2, 1]) / s
                q[2] = 0.25 * s
        return q

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
