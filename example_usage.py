"""
Example usage of the Robot class for object grasping tasks.

This script demonstrates how to:
1. Initialize the robot with camera and arm
2. Execute a grasping task loop
3. Properly cleanup resources
"""

from robot import Robot
import os

# ========== Configuration ==========

# Camera configuration
CAMERA_SERIAL_NUMBER = "your_camera_serial_number"  # Replace with your RealSense camera serial number
CAMERA_TO_BASE_PATH = 'real_world/calibration_result/camera_to_bases.pkl'

# Arm configuration
ARM_INTERFACE = "192.168.1.209"  # Robot arm IP address
INIT_POSE = [196.2, -1.6, 434, 179.2, 0, 0.3]  # Initial pose [X, Y, Z, roll, pitch, yaw]
INIT_SERVO_ANGLE = [0, -60, -30, 0, 90, 0]  # Initial servo angles
GRIPPER_ENABLE = True

# Vision module configuration
# You need to download these model weights first
GROUNDING_DINO_CONFIG = "path/to/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "path/to/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "path/to/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

# Task configuration
MAX_STEPS = 10  # Maximum number of execution steps
INSTRUCTION = "find all cubes"  # Object detection instruction


# ========== Main Script ==========

def main():
    """Main function to run the robot grasping task."""
    
    print("=" * 60)
    print("Robot Grasping Task Example")
    print("=" * 60)
    
    # Check if model paths exist
    if not os.path.exists(GROUNDING_DINO_CONFIG):
        print(f"\n⚠ Warning: Grounding DINO config not found at {GROUNDING_DINO_CONFIG}")
        print("Please update the path to your Grounding DINO config file.")
        return
    
    if not os.path.exists(GROUNDING_DINO_CHECKPOINT):
        print(f"\n⚠ Warning: Grounding DINO checkpoint not found at {GROUNDING_DINO_CHECKPOINT}")
        print("Please update the path to your Grounding DINO checkpoint file.")
        return
    
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"\n⚠ Warning: SAM checkpoint not found at {SAM_CHECKPOINT}")
        print("Please update the path to your SAM checkpoint file.")
        return
    
    # Initialize robot
    try:
        robot = Robot(
            serial_number=CAMERA_SERIAL_NUMBER,
            camera_to_base_path=CAMERA_TO_BASE_PATH,
            arm_interface=ARM_INTERFACE,
            init_pose=INIT_POSE,
            init_servo_angle=INIT_SERVO_ANGLE,
            gripper_enable=GRIPPER_ENABLE,
            grounding_dino_config_path=GROUNDING_DINO_CONFIG,
            grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            sam_checkpoint_path=SAM_CHECKPOINT,
            sam_model_type=SAM_MODEL_TYPE
        )
        
        # Execute grasping task
        robot.execute(max_steps=MAX_STEPS, instruction=INSTRUCTION)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup resources
        if 'robot' in locals():
            robot.cleanup()
    
    print("\n" + "=" * 60)
    print("Task completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

