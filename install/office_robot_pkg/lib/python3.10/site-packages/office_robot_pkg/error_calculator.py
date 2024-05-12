import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

class PositionErrorCalculator(Node):
    def __init__(self):
        super().__init__('position_error_calculator')
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.error_publisher = self.create_publisher(Float64, '/pos_error', 10)

        self.last_pose = None
        self.last_odom = None

        self.timer = self.create_timer(10.0, self.timer_callback)

    def pose_callback(self, msg):
        self.last_pose = msg.pose.pose

    def odom_callback(self, msg):
        self.last_odom = msg.pose.pose

    def timer_callback(self):
        self.calculate_error()

    def calculate_error(self):
        if self.last_pose is None or self.last_odom is None:
            return

        pose_position = self.last_pose.position
        odom_position = self.last_odom.position

        error_x = abs(pose_position.x - odom_position.x)
        error_y = abs(pose_position.y - odom_position.y)
        error_z = abs(pose_position.z - odom_position.z)
        total_error = (error_x**2 + error_y**2 + error_z**2)**0.5

        # self.get_logger().info(f'Position Error - X: {error_x}, Y: {error_y}, Z: {error_z}, Total: {total_error}')
        error_msg = Float64()
        error_msg.data = total_error
        self.error_publisher.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PositionErrorCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
