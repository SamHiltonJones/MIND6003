import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

def load_data(filepath, additional_filter=None, additional_filter2=None):
    data = np.load(filepath)
    data = data[~np.isinf(data).any(axis=1)] 
    if additional_filter is not None:
        data = data[additional_filter(data)]
    if additional_filter2 is not None:
        data = data[additional_filter2(data)]
    return data

def apply_transformation(data, rotation_matrix, translation_vector):
    translation_vector[2] = 0
    return np.dot(data, rotation_matrix.T) + translation_vector

def filter_data(data, z_threshold=1):
    return data[data[:, 2] > z_threshold]

def find_nearest_neighbors(data1, data2):
    tree1 = cKDTree(data1)
    _, indices = tree1.query(data2)
    return indices

def estimate_transformation(data1, data2, indices):
    translation = np.mean(data1[indices] - data2, axis=0)
    translation[2] = 0  
    rotation = np.eye(3)  
    return rotation, translation

def generate_random_transformations(num_transforms, translation_scale=10):
    translations = [translation_scale * (np.random.rand(3) - 0.5) for _ in range(num_transforms)]
    for trans in translations:
        trans[2] = 0  
    return translations

def calculate_loss(data1, data2):
    tree1 = cKDTree(data1)
    distances, _ = tree1.query(data2)
    return np.mean(distances)

def find_differences_and_clusters(array1, array2, threshold=0.3, min_samples=10, eps=0.5):
    tree1 = cKDTree(array1)
    tree2 = cKDTree(array2)
    
    distances2, _ = tree1.query(array2)
    significant_diff2 = array2[distances2 > threshold]
    
    if len(significant_diff2) > 0:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(significant_diff2)
        labels = db.labels_
        significant_clusters = significant_diff2[labels >= 0]
    else:
        print("No significant changes detected, nothing to cluster.")
        significant_clusters = np.array([])
    
    return significant_diff2, significant_clusters

def visualise_point_clouds(array1, array2, significant_diff, significant_clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='red', alpha=0.5, label='Array 1')
    # ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], c='green', alpha=0.5, label='Array 2')
    ax.scatter(significant_diff[:, 0], significant_diff[:, 1], significant_diff[:, 2], c='blue', label='Significant Differences')
    if significant_clusters.size > 0:
        ax.scatter(significant_clusters[:, 0], significant_clusters[:, 1], significant_clusters[:, 2], c='yellow', s=50, label='Significant Structures')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('3D Scatter Plot of Two Point Clouds - Differences and Clusters')
    plt.legend()
    plt.show()


def main():
    array1 = load_data('/home/sam/tut/MIND6003/whole_map_point_cloud.npy')
    array2 = load_data('point_cloud_data.npy', additional_filter=lambda x: x[:, 2] < 6, additional_filter2=lambda x: x[:, 2] < 3)
    
    num_samples = int(0.5 * len(array1))
    indices = np.random.choice(len(array1), num_samples, replace=False)
    array1 = array1[indices]
    
    num_samples = int(0.03 * len(array2))
    indices = np.random.choice(len(array2), num_samples, replace=False)
    array2 = array2[indices]
    
    array1_above = filter_data(array1)
    array2_above = filter_data(array2)
    
    num_transforms = 10 
    best_loss = np.inf
    best_transformation = None
    
    for translation in generate_random_transformations(num_transforms):
        for _ in range(10):  
            transformed_array2 = apply_transformation(array2_above, np.eye(3), translation)
            indices = find_nearest_neighbors(array1_above, transformed_array2)
            rotation, estimated_translation = estimate_transformation(array1_above, transformed_array2, indices)
            
            translation += estimated_translation
            transformed_array2 = apply_transformation(array2_above, rotation, translation)
            
            current_loss = calculate_loss(array1_above, transformed_array2)
            if current_loss < best_loss:
                best_loss = current_loss
                best_transformation = (rotation, translation)
    
    print("Best transformation:", best_transformation)
    print("Minimum loss:", best_loss)
    
    array2 = apply_transformation(array2, *best_transformation)

    significant_diff, significant_clusters = find_differences_and_clusters(array1, array2, threshold = 0.3)

    visualise_point_clouds(array1, array2, significant_diff, significant_clusters)

    whole_map = load_data('whole_map_point_cloud.npy')
    updated_whole_map = np.vstack([whole_map, significant_clusters])
    np.save('updated_whole_map_point_cloud.npy', updated_whole_map)
    visualise_point_clouds(array1, array2, significant_diff, significant_clusters)

if __name__ == "__main__":
    main()