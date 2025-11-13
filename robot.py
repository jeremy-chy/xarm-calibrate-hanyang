import time
import pickle
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import cv2
import re
from real_world.xarm6 import XARM6
from vision_module import VisionModule
from era_client import ERAClient


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
        sam_model_type="vit_h",
        era_server_url="http://127.0.0.1:5050",
        era_world_bounds=None,  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in meters
        coordinate_mode="discrete"  # "discrete" or "scaled"
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
            era_server_url: URL of the ERA model server
            era_world_bounds: Bounds of the workspace in era_world frame (meters)
                             [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            coordinate_mode: Mode for coordinate conversion
                            "discrete": continuous -> discrete int [0,100] -> continuous
                            "scaled": continuous * 100 (float) -> model output -> / 100
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

        # Load era_world matrices (camera_to_era_world, era_world_to_base)
        try:
            era_world_dict = pickle.load(open('era-world.pkl', 'rb'))
            camera_to_era_world, era_world_to_base = era_world_dict[serial_number]
            self.camera_to_era_world = camera_to_era_world
            self.era_world_to_base = era_world_to_base
            print("Loaded era_world transformation matrices successfully.")
        except Exception as e:
            print(f"Warning: Unable to load era_world transformations: {e}")
            self.camera_to_era_world = None
            self.era_world_to_base = None
        
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
        
        # Initialize ERA client for LLM-based policy
        print(f"Initializing ERA client (server: {era_server_url})...")
        self.era_client = ERAClient(server_url=era_server_url)
        print("ERA client initialized successfully.")
        
        # Set era_world bounds for coordinate mapping
        # Default: reasonable workspace bounds in meters
        if era_world_bounds is None:
            self.era_world_bounds = [[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.4]]  # meters
        else:
            self.era_world_bounds = era_world_bounds
        print(f"ERA world bounds set to: {self.era_world_bounds}")
        
        # Set coordinate mode
        if coordinate_mode not in ["discrete", "scaled"]:
            raise ValueError(f"coordinate_mode must be 'discrete' or 'scaled', got '{coordinate_mode}'")
        self.coordinate_mode = coordinate_mode
        print(f"Coordinate mode: {self.coordinate_mode}")
        
        # Initialize interaction history for policy
        self.interaction_history = []
        
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
    
    def camera_to_era_world_transform(self, points_camera):
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
        points_era_world = []
        # For display: show transform to era-world and from era-world to base as well
        for i, point_camera in enumerate(points_camera):
            # Convert to homogeneous coordinates
            point_homogeneous = np.append(point_camera, 1)  # Shape: (4,)

            # Camera -> Base (original)
            point_base_h = np.dot(self.camera_to_base_matrix, point_homogeneous)
            point_base = point_base_h[:3]

            # Camera -> Era-World
            if self.camera_to_era_world is not None and self.era_world_to_base is not None:
                point_era_world_h = np.dot(self.camera_to_era_world, point_homogeneous)
                point_era_world = point_era_world_h[:3]
                points_era_world.append(point_era_world)

        return points_era_world
    
    def era_world_to_discrete(self, points_era_world):
        """
        Convert era_world coordinates (meters) to discrete [0, 100] coordinates.
        Mode: "discrete" - rounds to integers
        
        Args:
            points_era_world: List of 3D points in era_world coordinates (meters)
        
        Returns:
            List of discrete 3D coordinates [X, Y, Z] in range [0, 100]
        """
        discrete_coords = []
        for point in points_era_world:
            discrete_point = []
            for i, val in enumerate(point):
                min_val, max_val = self.era_world_bounds[i]
                # Map [min_val, max_val] to [0, 100]
                discrete_val = int(round((val - min_val) / (max_val - min_val) * 100))
                discrete_val = np.clip(discrete_val, 0, 100)
                discrete_point.append(discrete_val)
            discrete_coords.append(discrete_point)
        return discrete_coords
    
    def era_world_to_scaled(self, points_era_world):
        """
        Convert era_world coordinates (meters) to scaled [0, 100] coordinates.
        Mode: "scaled" - keeps as floats
        
        Args:
            points_era_world: List of 3D points in era_world coordinates (meters)
        
        Returns:
            List of scaled 3D coordinates [X, Y, Z] in range [0, 100] (floats)
        """
        scaled_coords = []
        for point in points_era_world:
            scaled_point = []
            for i, val in enumerate(point):
                min_val, max_val = self.era_world_bounds[i]
                # Map [min_val, max_val] to [0, 100], keep as float
                scaled_val = (val - min_val) / (max_val - min_val) * 100
                scaled_val = np.clip(scaled_val, 0.0, 100.0)
                scaled_point.append(scaled_val)
            scaled_coords.append(scaled_point)
        return scaled_coords
    
    def discrete_to_era_world(self, discrete_coords):
        """
        Convert discrete [0, 100] coordinates to era_world coordinates (meters).
        Works for both discrete (int) and scaled (float) modes.
        
        Args:
            discrete_coords: List or array of length 3 with discrete/scaled coordinates [X, Y, Z]
        
        Returns:
            numpy array of 3D point in era_world coordinates (meters)
        """
        continuous_point = []
        for i, discrete_val in enumerate(discrete_coords[:3]):  # Only X, Y, Z
            min_val, max_val = self.era_world_bounds[i]
            # Map [0, 100] to [min_val, max_val]
            continuous_val = min_val + (discrete_val / 100.0) * (max_val - min_val)
            continuous_point.append(continuous_val)
        return np.array(continuous_point)
    
    def transform_era_world_to_base(self, point_era_world):
        """
        Transform a point from era_world coordinates to base coordinates.
        
        Args:
            point_era_world: 3D point in era_world coordinates (meters)
        
        Returns:
            3D point in base coordinates (meters)
        """
        if self.era_world_to_base is None:
            raise RuntimeError("era_world_to_base transformation not available")
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point_era_world, 1)
        
        # Apply transformation
        point_base_h = np.dot(self.era_world_to_base, point_homogeneous)
        
        return point_base_h[:3]
    
    def policy(self, step, color_image, points_era_world, instruction="Pick up the object"):
        """
        LLM-based policy function to determine the next action.
        
        Args:
            step: Current step number (1-indexed)
            color_image: Color image from camera (numpy array, RGB format)
            points_era_world: List of target object positions in era-world frame (meters)
            instruction: Task instruction string
        
        Returns:
            action: 4D vector [x, y, z, gripper_state] in millimeters
                   - (x, y, z): target position in base frame (mm)
                   - gripper_state: 1=open, 0=closed
        """
        # Step 1: Convert points_era_world to model coordinates based on mode
        if self.coordinate_mode == "discrete":
            # Discrete mode: round to integers [0, 100]
            model_coords = self.era_world_to_discrete(points_era_world)
            print(f"   Using discrete mode: {model_coords}")
        else:  # scaled
            # Scaled mode: keep as floats [0, 100]
            model_coords = self.era_world_to_scaled(points_era_world)
            # Format floats with 2 decimal places for cleaner display
            model_coords = [[round(x, 2) for x in coord] for coord in model_coords]
            print(f"   Using scaled mode: {model_coords}")
        
        # Step 2: Format object_info string
        # Format: {'object 1': [x, y, z], 'object 2': [x, y, z], ...}
        object_info_dict = {f"'object {i+1}'": coord for i, coord in enumerate(model_coords)}
        object_info = str(object_info_dict)
        
        # Step 3: Format interaction_history
        interaction_history = str(self.interaction_history)
        
        # Step 4: Send image directly as numpy array (will be base64 encoded by client)
        print(f"   Calling ERA model with:")
        print(f"   - Coordinate mode: {self.coordinate_mode}")
        print(f"   - Instruction: {instruction}")
        print(f"   - Object info: {object_info}")
        print(f"   - Interaction history: {interaction_history}")
        
        # Step 5: Call ERA client to get response
        # Pass numpy array directly, client will handle base64 encoding
        response = self.era_client.send_manipulation_request(
            images=[color_image],  # Pass numpy array directly
            instruction=instruction,
            object_info=object_info,
            interaction_history=interaction_history,
            encode_to_base64=True  # Enable base64 encoding for remote server
        )
        
        print(f"   ERA model response: {response}")
        
        # Step 6: Parse response to extract action
        # Expected format: <|action_start|>[X, Y, Z, Roll, Pitch, Yaw, Gripper]<|action_end|>
        action_parsed = self._parse_era_response(response)
        
        if action_parsed is None:
            raise ValueError("Failed to parse action from ERA model response")
        
        print(f"   Parsed model action: {action_parsed}")
        
        # Step 7: Convert model action to continuous era_world coordinates
        # action_parsed = [X, Y, Z, Roll, Pitch, Yaw, Gripper]
        position_model = action_parsed[:3]
        gripper_state = action_parsed[6]
        
        # Convert back to era_world (works for both modes)
        position_era_world = self.discrete_to_era_world(position_model)
        print(f"   Era-world position (m): {position_era_world}")
        
        # Step 8: Transform to base coordinates
        position_base = self.transform_era_world_to_base(position_era_world)
        print(f"   Base position (m): {position_base}")
        
        # Step 9: Convert to millimeters
        position_base_mm = position_base * 1000
        
        # Step 10: Construct 4D action vector
        action = np.array([
            position_base_mm[0],
            position_base_mm[1],
            position_base_mm[2],
            gripper_state
        ])
        
        # Step 11: Update interaction history
        # self.interaction_history.append(action_parsed)
        
        print(f"   Policy (step {step}): Action [{action[0]:.1f}, {action[1]:.1f}, {action[2]:.1f}, {action[3]}] mm")
        
        return action
    
    def _parse_era_response(self, response):
        """
        Parse ERA model response to extract action vector.
        
        Args:
            response: String response from ERA model
        
        Returns:
            List of 7 numbers [X, Y, Z, Roll, Pitch, Yaw, Gripper]
            In discrete mode: integers
            In scaled mode: can be floats
            Returns None if parsing fails
        """
        # Look for action between <|action_start|> and <|action_end|>
        pattern = r'<\|action_start\|>(.*?)<\|action_end\|>'
        match = re.search(pattern, response, re.DOTALL)
        
        if not match:
            print("   Warning: Could not find action tags in response")
            return None
        
        action_str = match.group(1).strip()
        
        # Remove brackets and parse numbers
        action_str = action_str.strip('[]')
        
        try:
            # Split by comma and convert to numbers
            if self.coordinate_mode == "discrete":
                # Discrete mode: convert to integers
                action_values = [int(float(x.strip())) for x in action_str.split(',')]
            else:  # scaled
                # Scaled mode: keep as floats for X, Y, Z; int for Roll, Pitch, Yaw, Gripper
                parts = [x.strip() for x in action_str.split(',')]
                action_values = []
                for i, part in enumerate(parts):
                    val = float(part)
                    # Keep X, Y, Z as floats; convert Roll, Pitch, Yaw, Gripper to ints
                    if i < 3:  # X, Y, Z
                        action_values.append(val)
                    else:  # Roll, Pitch, Yaw, Gripper
                        action_values.append(int(val))
            
            if len(action_values) != 7:
                print(f"   Warning: Expected 7 values, got {len(action_values)}")
                return None
            
            return action_values
        except Exception as e:
            print(f"   Warning: Failed to parse action values: {e}")
            return None


    
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
    
    def execute(self, max_steps=100, vision_instruction="find all cubes", 
                task_instruction="Pick up the object"):
        """
        Execute the grasping task in a loop.
        
        Args:
            max_steps: Maximum number of steps to execute
            vision_instruction: Text instruction for object detection (e.g., "find all cubes")
            task_instruction: Text instruction for the policy/task (e.g., "Pick up the red cube")
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
            print(f"2. Getting object masks from vision module (instruction: '{vision_instruction}')...")
            masks, boxes, scores = self.get_object_mask(color_frame, instruction=vision_instruction)
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
            
            # Step 4: Transform to era-world frame
            print("4. Transforming coordinates to era-world frame...")
            points_era_world = self.camera_to_era_world_transform(points_camera)
            print(f"   ✓ Transformed {len(points_era_world)} points to era-world frame")
            
            # Step 5: Execute action from policy
            print("5. Executing action from policy...")
            
            # Get color image as numpy array for policy
            color_image = np.asanyarray(color_frame.get_data())
            
            # Call policy to get the 4D action vector
            action = self.policy(step + 1, color_image, points_era_world, instruction=task_instruction)
            
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
            sam_model_type="vit_h",
            era_server_url="http://127.0.0.1:5050",  # ERA model server URL (can be remote)
            coordinate_mode="discrete"  # "discrete" (int) or "scaled" (float)
        )
        
        # Execute with LLM-based policy
        # vision_instruction: what to detect with the vision module
        # task_instruction: what task to perform (for the LLM policy)
        robot.execute(
            max_steps=5,
            vision_instruction="find all cubes",
            task_instruction="Pick up the red cube and place it on the blue cube"
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.cleanup()

