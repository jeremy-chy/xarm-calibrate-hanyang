import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud('real_pcd.pcd')
pcd.paint_uniform_color([1, 0, 0])
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, coordinate_frame])