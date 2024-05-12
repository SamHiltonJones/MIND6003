import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64
import subprocess

class VideoCapture(Node):
    def __init__(self):
        super().__init__('video_capture')
        self.video_subscriber = self.create_subscription(
            Int64, '/video_capture', self.video_callback, 5)
        self.process = None  

    def video_callback(self, msg):
        if msg.data == 1:
            self.start_script()
        elif msg.data == 0:
            self.stop_script()
            subprocess.run(['office_robot_pkg/office_robot_pkg/stop_video.py'], check=True)

    def start_script(self):
        if self.process is not None and self.process.poll() is None:
            self.get_logger().info('A script is already running, stopping it first.')
            self.stop_script()
        
        self.get_logger().info('Starting the video capture script.')
        self.process = subprocess.Popen(['office_robot_pkg/office_robot_pkg/video.py'])

    def stop_script(self):
        if self.process is not None:
            self.get_logger().info('Stopping the video capture script.')
            self.process.terminate()
            self.process.wait() 
            self.get_logger().info('Script stopped.')
        else:
            self.get_logger().info('No script is currently running.')

def main(args=None):
    rclpy.init(args=args)
    video_capture = VideoCapture()
    rclpy.spin(video_capture)
    video_capture.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
