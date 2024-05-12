import open3d as o3d
import numpy as np

def main():
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("output.ply")
    if pcd.is_empty():
        print("No data in PCD file.")
    else:
        print("Displaying point cloud with", len(pcd.points), "points.")
        np_points = np.asarray(pcd.points)
        print("Point coordinates range:")
        print("X min:", np.min(np_points[:, 0]), "X max:", np.max(np_points[:, 0]))
        print("Y min:", np.min(np_points[:, 1]), "Y max:", np.max(np_points[:, 1]))
        print("Z min:", np.min(np_points[:, 2]), "Z max:", np.max(np_points[:, 2]))
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="PointCloud Visualization", width=800, height=600)
        vis.add_geometry(pcd)

        ctr = vis.get_view_control()
        ctr.set_front([-1, -0.5, -0.5]) 
        ctr.set_lookat([0, 0, 150])  
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.3)

        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    main()
