import pickle
import numpy as np
import cv2
import open3d as o3d

pcd_in_board_frame = o3d.io.read_point_cloud('real_pcd.pcd')

rvec_new, tvec_new = pickle.load(open('real_world/calibration_result1/rvecs.pkl', 'rb')), pickle.load(open('real_world/calibration_result1/tvecs.pkl', 'rb'))
board_to_camera_transforms = {}
for serial_number in rvec_new.keys():
    rvec = rvec_new[serial_number]
    tvec = np.squeeze(tvec_new[serial_number])
    R_mtx, _ = cv2.Rodrigues(rvec)
    # rvec and tvec represent board_to_camera transformation
    # We need camera_to_board (inverse) to transform points from camera frame to board frame
    board_to_camera = np.array([[R_mtx[0][0], R_mtx[0][1], R_mtx[0][2], tvec[0]],
                                [R_mtx[1][0], R_mtx[1][1], R_mtx[1][2], tvec[1]],
                                [R_mtx[2][0], R_mtx[2][1], R_mtx[2][2], tvec[2]],
                                [0, 0, 0, 1]])
    # Invert the transformation: camera_to_board = board_to_camera^(-1)
    board_to_camera_transforms[serial_number] = board_to_camera

camera_to_robot = pickle.load(open('real_world/calibration_result/camera_to_bases.pkl', 'rb'))


T_board_to_robot = {}
serial_number = list(board_to_camera_transforms.keys())[0]
for serial_number in board_to_camera_transforms.keys():
    T_board_to_robot[serial_number] = np.dot(board_to_camera_transforms[serial_number], camera_to_robot[serial_number])

pcd_in_robot_frame = o3d.geometry.PointCloud()
points = np.asarray(pcd_in_board_frame.points)
points_hom = np.hstack([points, np.ones((points.shape[0], 1))])  # Make points homogeneous
transformed_points = np.dot(points_hom, np.dot(board_to_camera_transforms[serial_number].T, camera_to_robot[serial_number].T))
pcd_in_robot_frame.points = o3d.utility.Vector3dVector(transformed_points[:, :3])
# o3d.io.write_point_cloud('real_pcd_in_robot_frame.pcd', pcd_in_robot_frame)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_in_robot_frame, coordinate_frame])
o3d.io.write_point_cloud('real_pcd_in_robot_frame.pcd', pcd_in_robot_frame)

original_board_to_robot_transform = np.dot(board_to_camera_transforms[serial_number].T, camera_to_robot[serial_number].T)
np.save('original_board_to_robot_transform.npy', original_board_to_robot_transform)