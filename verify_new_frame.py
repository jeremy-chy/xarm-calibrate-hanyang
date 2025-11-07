import time
import pickle
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import cv2

camera_to_bases_main = pickle.load(open('real_world/calibration_result/camera_to_bases.pkl', 'rb'))

rvec_new, tvec_new = pickle.load(open('real_world/calibration_result1/rvecs.pkl', 'rb')), pickle.load(open('real_world/calibration_result1/tvecs.pkl', 'rb'))
camera_to_bases_new = {}
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
    camera_to_bases_new[serial_number] = np.linalg.inv(board_to_camera).T

serial_numbers = list(camera_to_bases_main.keys())
print(serial_numbers)
# del serial_numbers[2]
# del serial_numbers[0]
pcds = []

for i, serial_number in enumerate(serial_numbers):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    try:
        pipeline.start(config)
    except:
        continue
    time.sleep(2)

    camera_depth_to_disparity = rs.disparity_transform(True)
    camera_disparity_to_depth = rs.disparity_transform(False)
    camera_spatial = rs.spatial_filter()
    camera_spatial.set_option(rs.option.filter_magnitude, 5)
    camera_spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
    camera_spatial.set_option(rs.option.filter_smooth_delta, 1)
    camera_spatial.set_option(rs.option.holes_fill, 1)
    camera_temporal = rs.temporal_filter()
    camera_temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
    camera_temporal.set_option(rs.option.filter_smooth_delta, 1)
    camera_threshold = rs.threshold_filter()
    camera_threshold.set_option(rs.option.min_distance, 0)
    camera_threshold.set_option(rs.option.max_distance, 1.5)

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    aligned_depth_frame = frames.get_depth_frame()
    aligned_color_frame = frames.get_color_frame()

    aligned_depth_frame = camera_depth_to_disparity.process(aligned_depth_frame)
    # aligned_depth_frame = camera_spatial.process(aligned_depth_frame)
    aligned_depth_frame = camera_temporal.process(aligned_depth_frame)
    aligned_depth_frame = camera_disparity_to_depth.process(aligned_depth_frame)
    aligned_depth_frame = camera_threshold.process(aligned_depth_frame)

    pc = rs.pointcloud()
    pc.map_to(aligned_color_frame)
    points = pc.calculate(aligned_depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
    vtx_filtered = vtx[vtx[:, 2] <= 1.5]
    color_data = np.asanyarray(aligned_color_frame.get_data())
    color_data = color_data.reshape(-1, 3)  # colors
    color_filtered = color_data[vtx[:, 2] <= 1.5]

    # Normalize colors to 0-1 without applying any tint
    color_filtered = np.array(color_filtered / 255.0)

    vtx_homogeneous = np.hstack((vtx_filtered, np.ones((vtx_filtered.shape[0], 1))))
    vtx_transformed_homogeneous = np.dot(vtx_homogeneous, camera_to_bases_new[serial_number])
    vtx_transformed = vtx_transformed_homogeneous[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx_transformed)
    pcd.colors = o3d.utility.Vector3dVector(color_filtered)
    pcds.append(pcd)

merged_pcd = o3d.geometry.PointCloud()
for pcd in pcds:
    merged_pcd.points.extend(pcd.points)
    merged_pcd.colors.extend(pcd.colors)

# Filter points
points = np.asarray(merged_pcd.points)
colors = np.asarray(merged_pcd.colors)
# mask = (points[:, 2] >= -0.1) & (points[:, 2] <= 0.5) & (points[:, 0] >= 0.17) & (points[:, 0] <= 0.9) & (points[:, 1] >= -0.40) & (points[:, 1] <= 0.55)
# filtered_points = points[mask]
# filtered_colors = colors[mask]

filtered_points = points
filtered_colors = colors

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(filtered_points)
merged_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.io.write_point_cloud('real_pcd.pcd', merged_pcd) # Saves in the original board frame
o3d.visualization.draw_geometries([merged_pcd, coordinate_frame])
