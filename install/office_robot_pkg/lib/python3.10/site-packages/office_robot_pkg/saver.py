from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import cv2
import rclpy

class MySubscriber(Node):
    def __init__(self):
        super().__init__('my_subscriber')
        self.subscription = None
        self.subscription = self.create_subscription(PointCloud2, '/depth_camera/points', self.callback, 5)      
        
    def callback(self, msg: PointCloud2):
        # self.get_logger().info('Received PointCloud2 message')
        
        points_numpy = self.convert_point_cloud_msg_to_numpy(msg)

        theta_z, theta_y, theta_x = -np.pi/2, 0, -np.pi/2
        Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        rotation_matrix = Rz @ Ry @ Rx
        translation_vector = np.array([0.4, 0.0, 0.5])

        points_numpy = self.apply_transformation(points_numpy, rotation_matrix, translation_vector)

        if points_numpy is not None:
            np.save('/home/sam/tut/MIND6003/point_cloud_data.npy', points_numpy)

    def convert_point_cloud_msg_to_numpy(self, msg: PointCloud2):
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)


        points_list = list(gen) 
        if points_list:
            points_numpy = np.array(points_list) 

            points_numpy = np.vstack([points_numpy[name] for name in points_numpy.dtype.names]).astype(np.float32).T
        else:
            points_numpy = np.array([], dtype=np.float32)

        return points_numpy
    
    def apply_transformation(self, data, rotation_matrix, translation_vector):
        return (data @ rotation_matrix.T) + translation_vector

def main():
    rclpy.init()
    subscriber = MySubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
