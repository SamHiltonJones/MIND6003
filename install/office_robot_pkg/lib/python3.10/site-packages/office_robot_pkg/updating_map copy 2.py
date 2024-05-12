import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from std_msgs.msg import String, Int64

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')
        self.subscription = self.create_subscription(
            String,
            'matrix_topic',
            self.matrix_callback,
            10
        )
        self.settings_subscription = self.create_subscription(
            Int64,
            '/settings',
            self.settings_callback,
            10
        )
        self.settings_value = None
        self.transformation_matrix = None

        self.timer = self.create_timer(5, self.timer_callback)
        self.initial_whole_map_path = '/home/sam/tut/MIND6003/whole_map_point_cloud.npy'
        self.point_cloud_path = 'install/office_robot_pkg/share/office_robot_pkg/point_cloud/filtered_pcds/odom_view_map.pcd' 

        self.whole_map = self.load_data(self.initial_whole_map_path)
        self.initial_whole_map = np.copy(self.whole_map)

    def timer_callback(self):

        if self.settings_value is None:
            self.get_logger().info('No setting avaliable yet')
            return
        if self.transformation_matrix is None:
            self.get_logger().info('No transformation matrix available yet')
            return
        
        # self.get_logger().info('Processing point cloud...')
        # numpy_array = self.load_pcd_to_numpy(self.point_cloud_path)
        # np.save('numpy_point_cloud', numpy_array)

        array1 = self.load_data('/home/sam/tut/MIND6003/install/office_robot_pkg/share/office_robot_pkg/point_cloud/filtered_pcds/filtered_point_cloud.npy')
        array2 = self.load_data('/home/sam/tut/MIND6003/point_cloud_data.npy', additional_filter=lambda x: x[:, 2] < 6, additional_filter2=lambda x: x[:, 2] < 3)

        num_samples = int(0.5 * len(array1))
        indices = np.random.choice(len(array1), num_samples, replace=False)
        array1 = array1[indices]

        num_samples = int(0.03 * len(array2))
        indices = np.random.choice(len(array2), num_samples, replace=False)
        array2 = array2[indices]

        rotation_matrix = self.transformation_matrix[:3, :3]
        translation_vector = self.transformation_matrix[:3, 3]

        array2 = self.apply_transformation2(array2, rotation_matrix, translation_vector)

        num_transforms = 30
        best_loss = np.inf
        best_transformation = None

        array1_above = self.filter_data(array1)
        array2_above = self.filter_data(array2)

        for translation in self.generate_random_transformations(num_transforms):
            for _ in range(30):
                transformed_array2 = self.apply_transformation(array2_above, np.eye(3), translation)
                indices = self.find_nearest_neighbors(array1_above, transformed_array2)
                rotation, estimated_translation = self.estimate_transformation(array1_above, transformed_array2, indices)
                
                translation += estimated_translation
                transformed_array2 = self.apply_transformation(array2_above, rotation, translation)
                
                current_loss = self.calculate_loss(array1_above, transformed_array2)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_transformation = (rotation, translation)

        self.get_logger().info(f'Best Transformation: {best_transformation}')
        self.get_logger().info(f'Minimum Loss: {best_loss}')

        if self.settings_value == 1:
            self.setting1(array1, array2, best_transformation, best_loss)

        elif self.settings_value == 2:
            self.setting2(array1, array2, best_transformation, best_loss)

        self.visualise_point_clouds3(array1, transformed_array2, self.whole_map)

    def matrix_callback(self, msg):
        try:
            cleaned_data = msg.data.translate(str.maketrans('', '', '[]\n ')).split(',')
            matrix_data = np.array(cleaned_data, dtype=np.float64).reshape(4, 4)
            self.transformation_matrix = matrix_data
            self.get_logger().info(f'Transformation matrix received: {self.transformation_matrix}')
        except Exception as e:
            self.get_logger().error(f'Failed to process transformation matrix: {str(e)}')

    def settings_callback(self, msg):
        try:
            self.settings_value = msg.data
            self.get_logger().info(f'Received settings value: {self.settings_value}')
        except Exception as e:
            self.get_logger().error(f'No settings value updated: {str(e)}')

    def setting1(self, array1, array2, best_transformation, best_loss):
        if best_loss < 0.2:
            transformed_array2 = self.apply_transformation(array2, *best_transformation)
            significant_diff, significant_clusters = self.find_differences_and_clusters(array1, transformed_array2)

            self.update_whole_map(significant_clusters)
            self.visualise_point_clouds(array1, transformed_array2, significant_diff, significant_clusters)
        else:
            self.get_logger().info(f'No proper alignment found')
            transformed_array2 = self.apply_transformation(array2, *best_transformation)

            self.visualise_point_clouds2(array1, transformed_array2)
    
    def setting2(self, array1, array2, best_transformation, best_loss):
        if best_loss < 0.2:
            transformed_array2 = self.apply_transformation(array2, *best_transformation)
            significant_diff1, significant_clusters1 = self.find_differences_and_clusters(array1, transformed_array2)
            significant_diff2, significant_clusters2 = self.find_differences_and_clusters(transformed_array2, array1)

            self.removal_from_map(significant_clusters2)
            self.update_whole_map(significant_clusters1)
            self.visualise_point_clouds(transformed_array2, array1, significant_diff2, significant_clusters2)
            self.visualise_point_clouds(array1, transformed_array2, significant_diff1, significant_clusters1)


        else:
            self.get_logger().info(f'No proper alignment found')
            transformed_array2 = self.apply_transformation(array2, *best_transformation)

            self.visualise_point_clouds2(array1, transformed_array2)


        self.visualise_point_clouds3(array1, transformed_array2, self.whole_map)

    def removal_from_map(self, significant_clusters):
        if significant_clusters.size == 0:
            self.get_logger().info('No significant clusters provided for removal.')
            return

        distance_threshold = 0.1  # Adjust this value as needed

        tree = cKDTree(self.whole_map)

        indices_to_remove = set()
        for point in significant_clusters:
            idx = tree.query_ball_point(point, distance_threshold)
            indices_to_remove.update(idx)

        indices_to_remove = sorted(list(indices_to_remove), reverse=True)
        for idx in indices_to_remove:
            self.whole_map = np.delete(self.whole_map, idx, axis=0)

        removed_count = len(indices_to_remove)
        self.get_logger().info(f'Removed {removed_count} points from the map.')

        self.visualise_point_clouds3(array1=None, array2=None, array3=self.whole_map)
        np.save('updated_whole_map_point_cloud.npy', self.whole_map)
        self.get_logger().info('Updated whole map saved to updated_whole_map_point_cloud.npy.')


    def visualise_point_clouds(self, array1, array2, significant_diff, significant_clusters):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='red', alpha=0.5, label='Array 1')
        ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], c='green', alpha=0.5, label='Array 2')
        ax.scatter(significant_diff[:, 0], significant_diff[:, 1], significant_diff[:, 2], c='blue', label='Significant Differences')
        if significant_clusters.size > 0:
            ax.scatter(significant_clusters[:, 0], significant_clusters[:, 1], significant_clusters[:, 2], c='yellow', s=50, label='Significant Clusters')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title('3D Scatter Plot of Two Point Clouds - Differences and Clusters')
        plt.legend()
        plt.show()

    def visualise_point_clouds2(self, array1, array2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='red', alpha=0.5, label='Array 1')
        ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], c='green', alpha=0.5, label='Array 2')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title('3D Scatter Plot of Two Point Clouds - Differences and Clusters')
        plt.legend()
        plt.show()

    def visualise_point_clouds3(self, array1, array2, array3):
        array3 = array3[~np.isinf(array3).any(axis=1) & 
                        (array3[:, 0] > -6) & (array3[:, 0] < 5) & 
                        (array3[:, 1] > -5) & (array3[:, 1] < 5)]
        num_samples = int(1.0 * len(array3))
        indices = np.random.choice(len(array3), num_samples, replace=False)
        array3 = array3[indices] 

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='red', alpha=0.5, label='Array 1')
        # ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], c='green', alpha=0.5, label='Array 2')        
        ax.scatter(array3[:, 0], array3[:, 1], array3[:, 2], c='blue', alpha=0.5, label='Array 3')        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title('3D Scatter Plot of Two Point Clouds - Differences and Clusters')
        plt.legend()
        plt.show()

    def load_pcd_to_numpy(self, filepath):
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points, dtype=np.float32)
        return points

    def load_data(self, filepath, additional_filter=None, additional_filter2=None):
        data = np.load(filepath)
        data = data[~np.isinf(data).any(axis=1)]
        if additional_filter is not None:
            data = data[additional_filter(data)]
        if additional_filter2 is not None:
            data = data[additional_filter2(data)]
        return data
    
    def apply_transformation(self, data, rotation_matrix, translation_vector):
        # Ensure the translation in z-coordinate is always zero
        translation_vector[2] = 0
        return np.dot(data, rotation_matrix.T) + translation_vector
    

    def apply_transformation2(self, data, rotation_matrix, translation_vector):
        # Log the matrix components to verify their correctness
        self.get_logger().info(f'Applying rotation: {rotation_matrix}')
        self.get_logger().info(f'Applying translation: {translation_vector}')
        transformed_data = np.dot(data, rotation_matrix.T) + translation_vector
        return transformed_data

    def filter_data(self, data, z_threshold=2):
        # Exclude points where the z-coordinate is 0
        return data[(data[:, 2] > z_threshold) & (data[:, 2] != 0)]

    def find_nearest_neighbors(self, data1, data2):
        tree1 = cKDTree(data1)
        _, indices = tree1.query(data2)
        return indices

    def estimate_transformation(self, data1, data2, indices):
        translation = np.mean(data1[indices] - data2, axis=0)
        # Prevent any translation in the z-direction
        translation[2] = 0
        rotation = np.eye(3)
        return rotation, translation

    def generate_random_transformations(self, num_transforms, translation_scale=100):
        translations = [translation_scale * (np.random.rand(3) - 0.5) for _ in range(num_transforms)]
        # Set z-coordinate of transformations to zero
        for trans in translations:
            trans[2] = 0
        return translations

    def calculate_loss(self, data1, data2):
        tree1 = cKDTree(data1)
        distances, _ = tree1.query(data2)
        return np.mean(distances)

    def find_differences_and_clusters(self, array1, array2, threshold=0.2, min_samples=10, eps=0.5):
        tree1 = cKDTree(array1)
        tree2 = cKDTree(array2)
        
        distances, _ = tree1.query(array2)
        significant_diff = array2[distances > threshold]
        
        if len(significant_diff) > 0:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(significant_diff)
            labels = db.labels_
            significant_clusters = significant_diff[labels >= 0]
        else:
            significant_clusters = np.array([])

        return significant_diff, significant_clusters

    def update_whole_map(self, significant_clusters):
        if significant_clusters.size > 0:
            self.whole_map = np.vstack([self.whole_map, significant_clusters])
            np.save('updated_whole_map_point_cloud.npy', self.whole_map)
        else:
            self.get_logger().info('No significant clusters to update.')

def main(args=None):
    rclpy.init(args=args)
    point_cloud_processor = PointCloudProcessor()
    rclpy.spin(point_cloud_processor)
    point_cloud_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
