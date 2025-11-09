import time
import pickle
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import cv2
from real_world.xarm6 import XARM6
from vision_module import VisionModule


class Robot:
    def __init__(
        self,
        serial_number,
        camera_to_base_path='real_world/calibration_result/camera_to_bases.pkl',
        arm_interface="192.168.1.209",
        init_pose=[196.2, -1.6, 434, 179.2, 0, 0.3],
        init_servo_angle=[0, -60, -30, 0, 90, 0],
        gripper_enable=True,
        grounding_dino_config_path=None,
        grounding_dino_checkpoint_path=None,
        sam_checkpoint_path=None,
        sam_model_type="vit_h"
    ):
        """
        Initialize the robot with camera and arm.
        
        Args:
            serial_number: RealSense camera serial number
            camera_to_base_path: Path to the calibration file
            arm_interface: IP address of the robot arm
            init_pose: Initial pose of the arm [X, Y, Z, roll, pitch, yaw]
            init_servo_angle: Initial servo angles
            gripper_enable: Whether to enable the gripper
            grounding_dino_config_path: Path to Grounding DINO config
            grounding_dino_checkpoint_path: Path to Grounding DINO checkpoint
            sam_checkpoint_path: Path to SAM checkpoint
            sam_model_type: SAM model type (vit_h, vit_l, vit_b)
        """
        print("Initializing Robot...")
        
        # Initialize arm
        print("Initializing arm...")
        self.arm = XARM6(
            interface=arm_interface,
            init_pose=init_pose,
            init_servo_angle=init_servo_angle,
            gripper_enable=gripper_enable
        )
        print("Arm initialized successfully.")
        
        # Initialize camera
        print(f"Initializing camera with serial number: {serial_number}")
        self.serial_number = serial_number
        self._init_camera()
        print("Camera initialized successfully.")
        
        # Load camera to base transformation
        print(f"Loading calibration from {camera_to_base_path}")
        camera_to_bases = pickle.load(open(camera_to_base_path, 'rb'))
        self.camera_to_base_matrix = camera_to_bases[serial_number]
        print("Calibration loaded successfully.")
        
        # Initialize vision module
        if all([grounding_dino_config_path, grounding_dino_checkpoint_path, sam_checkpoint_path]):
            print("Initializing vision module...")
            self.vision = VisionModule(
                grounding_dino_config_path=grounding_dino_config_path,
                grounding_dino_checkpoint_path=grounding_dino_checkpoint_path,
                sam_checkpoint_path=sam_checkpoint_path,
                sam_model_type=sam_model_type
            )
            print("Vision module initialized successfully.")
        else:
            print("Warning: Vision module not initialized. Please provide model paths.")
            self.vision = None
        
        print("Robot initialization complete!\n")
    
    def _init_camera(self):
        """Initialize RealSense camera with filters."""
        # Camera pipeline setup (lines 15-24 from verify_stationary_cameras.py)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        try:
            self.pipeline.start(config)
        except Exception as e:
            raise RuntimeError(f"Failed to start camera: {e}")
        time.sleep(2)  # Let camera stabilize
        
        # Setup depth filters (lines 26-38 from verify_stationary_cameras.py)
        self.camera_depth_to_disparity = rs.disparity_transform(True)
        self.camera_disparity_to_depth = rs.disparity_transform(False)
        self.camera_spatial = rs.spatial_filter()
        self.camera_spatial.set_option(rs.option.filter_magnitude, 5)
        self.camera_spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.camera_spatial.set_option(rs.option.filter_smooth_delta, 1)
        self.camera_spatial.set_option(rs.option.holes_fill, 1)
        self.camera_temporal = rs.temporal_filter()
        self.camera_temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.camera_temporal.set_option(rs.option.filter_smooth_delta, 1)
        self.camera_threshold = rs.threshold_filter()
        self.camera_threshold.set_option(rs.option.min_distance, 0)
        self.camera_threshold.set_option(rs.option.max_distance, 1.5)
        
        # Alignment object
        self.align = rs.align(rs.stream.color)
    
    def capture_frame(self):
        """
        Capture a frame from the camera.
        Based on lines 40-52 from verify_stationary_cameras.py
        
        Returns:
            aligned_depth_frame: Aligned and filtered depth frame
            aligned_color_frame: Aligned color frame
        """
        # Get frames from pipeline
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        # Align depth to color
        frames = self.align.process(frames)
        aligned_depth_frame = frames.get_depth_frame()
        aligned_color_frame = frames.get_color_frame()
        
        # Apply filters to improve depth quality
        aligned_depth_frame = self.camera_depth_to_disparity.process(aligned_depth_frame)
        # Skip spatial filter (commented out in original code)
        # aligned_depth_frame = self.camera_spatial.process(aligned_depth_frame)
        aligned_depth_frame = self.camera_temporal.process(aligned_depth_frame)
        aligned_depth_frame = self.camera_disparity_to_depth.process(aligned_depth_frame)
        aligned_depth_frame = self.camera_threshold.process(aligned_depth_frame)
        
        return aligned_depth_frame, aligned_color_frame
    
    def get_object_mask(self, color_frame, instruction="find all cubes", 
                        box_threshold=0.35, text_threshold=0.25):
        """
        Get 2D mask of target object from vision module.
        
        Args:
            color_frame: Color frame from camera (RealSense frame)
            instruction: Text instruction for object detection (e.g., "find all cubes")
            box_threshold: Box confidence threshold for Grounding DINO
            text_threshold: Text confidence threshold for Grounding DINO
            
        Returns:
            masks: List of 2D binary masks (numpy arrays), one for each detected object
            boxes: List of bounding boxes
            scores: List of confidence scores
        """
        if self.vision is None:
            raise RuntimeError("Vision module not initialized. Please provide model paths in __init__.")
        
        # Convert RealSense color frame to numpy array (RGB format)
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save to temporary file (vision module expects a file path)
        temp_image_path = "temp_frame.jpg"
        color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_image_path, color_bgr)
        
        # Call vision module to generate masks
        masks, boxes, scores = self.vision.generate_mask(
            image_path=temp_image_path,
            instruction=instruction,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        return masks, boxes, scores
    
    def mask_to_camera_3d(self, masks, depth_frame, color_frame):
        """
        Convert 2D masks to 3D coordinates in camera frame.
        Based on lines 54-61 from verify_stationary_cameras.py
        
        For each mask in the list, we:
        1. Generate the full point cloud from depth frame
        2. Filter points using the mask
        3. Calculate the centroid (average position) of the masked points
        
        Args:
            masks: List of 2D binary masks (numpy arrays)
            depth_frame: Depth frame (RealSense frame)
            color_frame: Color frame (RealSense frame)
            
        Returns:
            points_3d_list: List of 3D points (centroids) in camera coordinate system
                           Each point is a numpy array of shape (3,) representing [x, y, z]
        """
        # Generate point cloud from depth frame (lines 54-57)
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        
        # Get vertices as numpy array (N, 3) where N is number of pixels
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # Get color data (for debugging/visualization purposes)
        color_data = np.asanyarray(color_frame.get_data())
        h, w = color_data.shape[:2]
        
        points_3d_list = []
        
        # Process each mask
        for i, mask in enumerate(masks):
            # Flatten the mask to 1D (same shape as vtx)
            # mask is (H, W), we need to flatten it to (H*W,)
            mask_flat = mask.flatten()
            
            # Filter points using the mask
            # Only keep points where mask is True
            masked_points = vtx[mask_flat]
            
            # Remove invalid points (where z == 0, meaning no depth data)
            valid_mask = masked_points[:, 2] > 0
            masked_points_valid = masked_points[valid_mask]

            print(f"  Mask {i+1}: {len(masked_points_valid)} valid points")
            
            if len(masked_points_valid) == 0:
                print(f"Warning: Mask {i+1} has no valid 3D points. Skipping.")
                continue
            
            # Calculate centroid (average position)
            centroid = np.mean(masked_points_valid, axis=0)
            
            print(f"  Mask {i+1}: {len(masked_points_valid)} valid points, centroid at {centroid}")
            
            points_3d_list.append(centroid)
        
        return points_3d_list
    
    def camera_to_base_transform(self, points_camera):
        """
        Transform points from camera coordinate system to base coordinate system.
        Based on lines 66-68 from verify_stationary_cameras.py
        
        Args:
            points_camera: List of 3D points in camera coordinate system
                          Each point is a numpy array of shape (3,) representing [x, y, z]
            
        Returns:
            points_base: List of 3D points in base coordinate system
                        Each point is a numpy array of shape (3,) representing [x, y, z]
        """
        points_base = []
        
        for i, point_camera in enumerate(points_camera):
            # Convert to homogeneous coordinates (add 1 as the 4th element)
            point_homogeneous = np.append(point_camera, 1)  # Shape: (4,)
            
            # Apply transformation matrix
            # Note: camera_to_base_matrix is a 4x4 transformation matrix
            point_transformed_homogeneous = np.dot(self.camera_to_base_matrix, point_homogeneous)
            
            # Convert back to 3D coordinates (remove the homogeneous coordinate)
            point_base = point_transformed_homogeneous[:3]
            
            print(f"  Point {i+1}: Camera {point_camera} -> Base {point_base}")
            
            points_base.append(point_base)
        
        return points_base
    
    def policy(self, step, color_image, points_base):
        """
        Policy function to determine the next action.
        
        Args:
            step: Current step number (1-indexed)
            color_image: Color image from camera (numpy array, RGB format)
            points_base: List of target object positions in base frame (meters)
        
        Returns:
            action: 4D vector [x, y, z, gripper_state] in millimeters
                   - (x, y, z): target position in mm
                   - gripper_state: 1=open, 0=closed
        """
        if step == 1:
            # Step 1: Grasp the first object
            # Target position: first object, gripper closed
            if len(points_base) == 0:
                raise ValueError("No objects detected! Cannot execute policy.")
            
            target_position_m = points_base[0]  # meters
            target_position_mm = target_position_m * 1000  # convert to mm

            
            action = np.array([
                target_position_mm[0]-20,
                target_position_mm[1]-30,
                target_position_mm[2],
                0  # Gripper closed (grasp)
            ])
            
            # Save the grasp position for future steps
            self.last_grasp_position = target_position_mm.copy()
            
            print(f"   Policy (step {step}): Grasp first object at [{target_position_mm[0]:.1f}, {target_position_mm[1]:.1f}, {target_position_mm[2]:.1f}] mm")
            return action
        
        elif step == 2:
            # Step 2: Lift object by 20cm (200mm)
            if not hasattr(self, 'last_grasp_position'):
                raise ValueError("No previous grasp position found!")
            
            lift_offset = 200  # mm (20cm)
            action = np.array([
                self.last_grasp_position[0],
                self.last_grasp_position[1],
                self.last_grasp_position[2] + lift_offset,
                0  # Gripper remains closed
            ])
            
            print(f"   Policy (step {step}): Lift object by 20cm to [{action[0]:.1f}, {action[1]:.1f}, {action[2]:.1f}] mm")
            return action
        
        elif step == 3:
            # Step 3+: Maintain position (same as step 2)
            target_position_m = points_base[1]  # meters
            target_position_mm = target_position_m * 1000  # convert to mm

            
            action = np.array([
                target_position_mm[0]-20,
                target_position_mm[1]-40,
                target_position_mm[2] + 40, # for a stack operation
                1  # Gripper open (stack)
            ])
            
            # Save the grasp position for future steps
            self.last_grasp_position = target_position_mm.copy()
            
            print(f"   Policy (step {step}): Stack object at [{target_position_mm[0]:.1f}, {target_position_mm[1]:.1f}, {target_position_mm[2]:.1f}] mm")
            return action

        else:

            lift_offset = 200  # mm (20cm)
            action = np.array([
                self.last_grasp_position[0],
                self.last_grasp_position[1],
                self.last_grasp_position[2] + lift_offset,
                1  # Gripper remains closed
            ])
            
            print(f"   Policy (step {step}): Lift object by 20cm to [{action[0]:.1f}, {action[1]:.1f}, {action[2]:.1f}] mm")
            return action


    
    def grasp(self, target_vector):
        """
        Execute grasping motion.
        
        Args:
            target_vector: Length-4 vector [x, y, z, gripper_state]
                          - (x, y, z): Target position of gripper in base coordinates (mm)
                          - gripper_state: 1 for open, 0 for close
        """
        if len(target_vector) != 4:
            raise ValueError(f"target_vector must have length 4, got {len(target_vector)}")
        
        # Extract target position and gripper state
        target_x, target_y, target_z, target_gripper_state = target_vector
        
        # Step 1: Move to target position
        # Note: xarm's move_to_pose controls the flange center position,
        # but input is gripper position. Need to adjust z by gripper height (175mm)
        
        # Get current pose to preserve orientation (roll, pitch, yaw)
        current_pose = self.arm.get_current_pose()
        current_roll, current_pitch, current_yaw = current_pose[3], current_pose[4], current_pose[5]
        
        # Calculate flange position from gripper position
        # Gripper is 175mm below the flange center
        flange_x = target_x
        flange_y = target_y
        flange_z = target_z + 175.0  # Subtract gripper height
        
        # Construct target pose [X, Y, Z, roll, pitch, yaw]
        # Keep orientation unchanged
        target_pose = [flange_x, flange_y, flange_z, current_roll, current_pitch, current_yaw]
        
        print(f"   Moving to position: gripper=({target_x:.1f}, {target_y:.1f}, {target_z:.1f}), "
              f"flange=({flange_x:.1f}, {flange_y:.1f}, {flange_z:.1f})")
        
        # Move arm to target position
        self.arm.move_to_pose(pose=target_pose, wait=True, ignore_error=False)
        print(f"   ✓ Movement completed")
        
        # Step 2: Actuate gripper
        # Check current gripper state to avoid redundant operations
        current_gripper_position = self.arm.get_gripper_state()
        
        # Determine if gripper is currently open (threshold: > 400 means open)
        is_currently_open = current_gripper_position > 400
        target_is_open = target_gripper_state == 1
        
        if target_is_open and not is_currently_open:
            print(f"   Opening gripper (current position: {current_gripper_position})")
            self.arm.open_gripper(wait=True)
            print(f"   ✓ Gripper opened")
        elif not target_is_open and is_currently_open:
            print(f"   Closing gripper (current position: {current_gripper_position})")
            self.arm.close_gripper(wait=True)
            print(f"   ✓ Gripper closed")
        else:
            print(f"   Gripper already in target state (position: {current_gripper_position})")
        
        print(f"   ✓ Grasp operation completed")
    
    def execute(self, max_steps=100, instruction="find all cubes"):
        """
        Execute the grasping task in a loop.
        
        Args:
            max_steps: Maximum number of steps to execute
            instruction: Text instruction for object detection (e.g., "find all cubes")
        """
        print("Starting execution...")
        step = 0
        
        while step < max_steps:
            print(f"\n=== Step {step + 1} ===")
            
            # Step 1: Capture frame
            print("1. Capturing frame...")
            depth_frame, color_frame = self.capture_frame()
            print("   ✓ Frame captured successfully")
            
            # Step 2: Get object masks from vision module
            print(f"2. Getting object masks from vision module (instruction: '{instruction}')...")
            masks, boxes, scores = self.get_object_mask(color_frame, instruction=instruction)
            print(f"   ✓ Detected {len(masks)} objects")
            
            # Check if any objects were detected
            if len(masks) == 0:
                print("   ⚠ No objects detected. Skipping this step.")
                step += 1
                continue
            
            # Step 3: Convert 2D masks to 3D in camera frame
            print("3. Converting masks to 3D coordinates in camera frame...")
            points_camera = self.mask_to_camera_3d(masks, depth_frame, color_frame)
            print(f"   ✓ Extracted {len(points_camera)} valid 3D centroids")
            
            # Check if any valid points were extracted
            if len(points_camera) == 0:
                print("   ⚠ No valid 3D points extracted. Skipping this step.")
                step += 1
                continue
            
            # Step 4: Transform to base frame
            print("4. Transforming coordinates to base frame...")
            points_base = self.camera_to_base_transform(points_camera)
            print(f"   ✓ Transformed {len(points_base)} points to base frame")
            
            # Step 5: Execute action from policy
            print("5. Executing action from policy...")
            
            # Get color image as numpy array for policy
            color_image = np.asanyarray(color_frame.get_data())
            
            # Call policy to get the 4D action vector
            action = self.policy(step + 1, color_image, points_base)
            
            # Execute the action using grasp
            print(f"   Executing action: [{action[0]:.1f}, {action[1]:.1f}, {action[2]:.1f}, {action[3]}]")
            self.grasp(action)
            print(f"   ✓ Action executed successfully")
            
            print(f"\nStep {step + 1} completed.")
            step += 1
            
            # TODO: Add termination condition
            # - Break if task is complete
            # - Break if no objects detected
            # - Break on error
            
        print("\nExecution finished!")
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        print("Cleanup complete.")


if __name__ == "__main__":
    # Example usage
    serial_number = "311322303615"  # Replace with actual serial number
    
    try:
        robot = Robot(
            serial_number=serial_number,
            grounding_dino_config_path="/home/hanyang/Downloads/xarm-calibrate-hanyang/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounding_dino_checkpoint_path="/home/hanyang/Downloads/xarm-calibrate-hanyang/models/groundingdino_swint_ogc.pth",
            sam_checkpoint_path="/home/hanyang/Downloads/xarm-calibrate-hanyang/models/sam_vit_h_4b8939.pth",
            sam_model_type="vit_h"
        )
        robot.execute(max_steps=3)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.cleanup()

