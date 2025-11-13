import pickle
import numpy as np
import cv2

rvecs = pickle.load(open('real_world/calibration_result_unflip_era_world/rvecs.pkl', 'rb'))
tvecs = pickle.load(open('real_world/calibration_result_unflip_era_world/tvecs.pkl', 'rb'))

camera_to_bases = pickle.load(open('real_world/calibration_result/camera_to_bases.pkl', 'rb'))

serial_numbers = list(rvecs.keys())

print(serial_numbers)

era_world = {}

for serial_number in serial_numbers:
    rvec = rvecs[serial_number]
    tvec = np.squeeze(tvecs[serial_number])
    R_mtx, _ = cv2.Rodrigues(rvec)
    world_to_camera = np.array([[R_mtx[0][0], R_mtx[0][1], R_mtx[0][2], tvec[0]],
                      [R_mtx[1][0], R_mtx[1][1], R_mtx[1][2], tvec[1]],
                      [R_mtx[2][0], R_mtx[2][1], R_mtx[2][2], tvec[2]],
                      [0, 0, 0, 1]])
    world_to_era_world = np.array([[-1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
    camera_to_world = np.linalg.inv(world_to_camera)
    camera_to_era_world = np.dot(world_to_era_world, camera_to_world)

    camera_to_base_matrix = camera_to_bases[serial_number]
    era_world_to_base = np.dot(camera_to_base_matrix, np.linalg.inv(camera_to_era_world))

    era_world[serial_number] = (camera_to_era_world, era_world_to_base)

    # Save the entire era_world dictionary as a pickle file
with open('era-world.pkl', 'wb') as f:
    pickle.dump(era_world, f)

