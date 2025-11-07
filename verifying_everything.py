import numpy as np
import open3d as o3d

original_board_to_robot_transform = np.load('original_board_to_robot_transform.npy')
pcd_in_robot_frame = o3d.io.read_point_cloud('real_pcd_in_robot_frame.pcd')
pcd_in_original_board_frame = pcd_in_robot_frame.transform(np.linalg.inv(original_board_to_robot_transform).T)

original_to_flipped_board_transform = np.array([[-1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, -1, 0],
                                                [0, 0, 0, 1]])
pcd_in_flipped_board_frame = pcd_in_original_board_frame.transform(original_to_flipped_board_transform)

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_in_flipped_board_frame, coordinate_frame])