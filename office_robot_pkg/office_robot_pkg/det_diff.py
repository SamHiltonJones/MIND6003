import rclpy
from time import sleep
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DynDetection(Node):
    def __init__(self):
        super().__init__('DynDetection')
        self.cam_subscriber = self.create_subscription(
            Image,
            '/depth_camera/image_raw',
            self.image_callback,
            10)

        self.subscription = self.create_subscription(
            PointCloud2,
            '/depth_camera/points',
            self.pc_callback,
            10)

        self.previous_point_cloud = None
        self.current_point_cloud = None
        self.previous_point_cloud_data = None
        self.current_point_cloud_data = None

        self.timer = self.create_timer(2, self.detect_movement)

    def pc_callback(self, msg):
        if self.current_point_cloud_data is not None:
            self.previous_point_cloud_data = self.current_point_cloud_data
            np.save('previous_point_cloud.npy', np.array(self.previous_point_cloud_data).reshape(-1, 4))
        
        self.current_point_cloud = msg
        self.current_point_cloud_data = msg.data
        np.save('current_point_cloud.npy', np.array(self.current_point_cloud_data).reshape(-1, 4))
        # self.detect_movement()

    def image_callback(self, msg):
        pass

    def detect_movement(self):
        self.get_logger().info("Detecting.............")
        if self.previous_point_cloud is not None and len(np.asarray(self.previous_point_cloud_data)) > 0 and self.current_point_cloud is not None:
            previous_pc_np_data = np.array(self.previous_point_cloud_data).astype(np.int64).reshape(-1, 4)
            current_pc_np_data = np.array(self.current_point_cloud_data).astype(np.int64).reshape(-1, 4)
            mse = np.mean((previous_pc_np_data - current_pc_np_data) ** 2)

            threshold = 620
            movement_detected = np.any(mse > threshold)
            if movement_detected:
                self.get_logger().info("Motion Detected!!")
                self.get_logger().info(str(mse))
                ds_current_pc_np = self.random_downsampling(current_pc_np_data, 0.005)
                ds_previous_pc_np = self.random_downsampling(previous_pc_np_data, 0.005)
                diff = ds_current_pc_np - ds_previous_pc_np
                diff = np.delete(diff, 3, axis=1)
                self.create_scatt(diff)
            else:
                self.get_logger().info("Pointclouds seem identical, no movement")
                self.get_logger().info(str(mse))

            self.previous_point_cloud = self.current_point_cloud
            self.previous_point_cloud_data = self.current_point_cloud_data

    def create_scatt(self, diff):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(diff[:, 0], diff[:, 1], diff[:, 2], c=diff[:, 2], cmap='viridis')
        ax.set_xlabel('X Difference')
        ax.set_ylabel('Y Difference')
        ax.set_zlabel('Z Difference')
        ax.set_title('3D Scatter Plot of Point Cloud Differences')
        plt.savefig("scatterplot_filename.png")

    def random_downsampling(self, point_cloud, ratio):
        num_points = len(point_cloud)
        num_points_to_keep = int(num_points * ratio)
        if num_points_to_keep >= num_points:
            return point_cloud
        else:
            indices = np.random.choice(num_points, num_points_to_keep, replace=False)
            return point_cloud[indices]


def main(args=None):
    rclpy.init(args=args)
    dyn_detect = DynDetection()
    rclpy.spin(dyn_detect)
    dyn_detect.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
