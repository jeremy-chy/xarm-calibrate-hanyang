import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import argparse

def capture_image(save_path, serial_number=None):
    """
    Initializes the RealSense camera, captures a single frame, and saves it as a JPG file.
    
    Args:
        save_path (str): The path where the image will be saved (e.g., 'images/photo.jpg').
        serial_number (str, optional): The serial number of the camera. If None, uses the first available.
    """
    print(f"Initializing camera{f' {serial_number}' if serial_number else ''}...")
    
    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    if serial_number:
        config.enable_device(serial_number)
    
    # Enable color stream
    # Using 640x480 to match robot.py configuration
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    try:
        # Start pipeline
        pipeline.start(config)
        
        # Warm up the camera
        print("Waiting for camera to stabilize (2 seconds)...")
        time.sleep(2)
        
        # Capture frames
        print("Capturing frame...")
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("Error: Could not get color frame.")
            return False
            
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert RGB to BGR (for OpenCV)
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save the image
        success = cv2.imwrite(save_path, color_image_bgr)
        
        if success:
            print(f"Successfully saved image to: {save_path}")
        else:
            print(f"Failed to save image to: {save_path}")
            
        return success
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        # Stop pipeline
        print("Closing camera pipeline...")
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a single image from RealSense camera.")
    parser.add_argument("save_path", type=str, help="Path to save the captured image (e.g., ./test.jpg)")
    parser.add_argument("--serial", type=str, default=None, help="Serial number of the RealSense camera (optional)")
    
    args = parser.parse_args()
    
    capture_image(args.save_path, args.serial)

