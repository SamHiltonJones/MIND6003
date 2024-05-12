from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct

def callback(self, ros_point_cloud):
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            test = x[3] 
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)

            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)

        out_pcd = o3d.geometry.PointCloud()    
        out_pcd.points = o3d.utility.Vector3dVector(xyz)
        out_pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud("cloud_data.ply",out_pcd)