# Robot Control with ERA Model - Usage Guide

## Overview

This guide explains the new features added to the robot control system:
1. **Base64 Image Encoding**: Images are now sent over network using base64 encoding (client and server can be on different machines)
2. **Coordinate Mode Selection**: Two modes for coordinate conversion - "discrete" and "scaled"

## Feature 1: Remote Image Transfer via Base64

### What Changed?
- Previously: Images were passed as file paths (only worked locally)
- Now: Images are encoded to base64 and sent over HTTP (works remotely)

### How It Works

#### In `era_client.py`:
```python
# Images can now be:
# - File paths (str)
# - Numpy arrays (np.ndarray)  
# - PIL Images (Image.Image)

client.send_manipulation_request(
    images=[color_image],  # Can be any of the above types
    encode_to_base64=True  # Enable base64 encoding (default: True)
)
```

#### In `era_server.py`:
The server automatically detects and decodes base64 images:
- Data URI format: `data:image/jpeg;base64,/9j/4AAQ...`
- Plain base64 string
- File paths (for backward compatibility)
- URLs

### Usage Example
```python
# Client on Machine A
robot = Robot(
    serial_number="311322303615",
    era_server_url="http://192.168.1.100:5050",  # Remote server
    ...
)

# Images are automatically base64-encoded and sent to the server
robot.execute(...)
```

## Feature 2: Coordinate Conversion Modes

### Two Modes

#### Mode 1: "discrete" (Default)
- Continuous coordinates → **Integer** [0, 100] → Model → **Integer** output → Continuous
- Example: `[0.15, -0.22, 0.08]` meters → `[75, 13, 20]` discrete → Model outputs `[80, 15, 22]` → `[0.16, -0.20, 0.09]` meters

#### Mode 2: "scaled"
- Continuous coordinates → **Float** [0, 100] → Model → **Float** output → Continuous  
- Example: `[0.15, -0.22, 0.08]` meters → `[75.5, 13.2, 20.3]` scaled → Model outputs `[80.7, 15.1, 22.5]` → `[0.161, -0.198, 0.090]` meters

### When to Use Each Mode?

**Use "discrete" when:**
- The model was trained with discrete integer coordinates
- You want simpler, more interpretable coordinates
- You don't need sub-centimeter precision

**Use "scaled" when:**
- The model can handle continuous values
- You need finer precision (sub-centimeter)
- The model outputs float coordinates

### Usage Example

```python
# Discrete mode (integers)
robot = Robot(
    serial_number="311322303615",
    coordinate_mode="discrete",  # Default
    era_world_bounds=[[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.4]],  # meters
    ...
)

# Object info sent to model:
# {'object 1': [73, 15, 18], 'object 2': [57, 20, 18], ...}
```

```python
# Scaled mode (floats)
robot = Robot(
    serial_number="311322303615",
    coordinate_mode="scaled",
    era_world_bounds=[[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.4]],  # meters
    ...
)

# Object info sent to model:
# {'object 1': [73.5, 15.2, 18.7], 'object 2': [57.3, 20.1, 18.4], ...}
```

## Complete Example

```python
from robot import Robot

# Initialize robot with both features
robot = Robot(
    serial_number="311322303615",
    
    # Vision module paths
    grounding_dino_config_path="path/to/config.py",
    grounding_dino_checkpoint_path="path/to/checkpoint.pth",
    sam_checkpoint_path="path/to/sam.pth",
    sam_model_type="vit_h",
    
    # ERA model server (can be remote!)
    era_server_url="http://192.168.1.100:5050",
    
    # Workspace bounds in meters
    era_world_bounds=[[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.4]],
    
    # Coordinate mode
    coordinate_mode="scaled",  # or "discrete"
)

# Execute task
robot.execute(
    max_steps=5,
    vision_instruction="find all cubes",
    task_instruction="Pick up the red cube and place it on the blue cube"
)

robot.cleanup()
```

## Technical Details

### Coordinate Conversion Functions

#### Discrete Mode:
- `era_world_to_discrete()`: meters → integers [0, 100]
- `discrete_to_era_world()`: integers [0, 100] → meters

#### Scaled Mode:
- `era_world_to_scaled()`: meters → floats [0, 100]
- `discrete_to_era_world()`: floats [0, 100] → meters (same function works for both!)

### Image Encoding
- Format: JPEG
- Encoding: Base64
- Prefix: `data:image/jpeg;base64,`
- Client automatically handles encoding
- Server automatically detects and decodes

## Troubleshooting

### Issue: Model returns integers but coordinate_mode is "scaled"
**Solution**: The parser in `_parse_era_response()` automatically handles both:
- In discrete mode: converts to `int`
- In scaled mode: keeps X, Y, Z as `float`, converts Roll, Pitch, Yaw, Gripper to `int`

### Issue: Images not displaying on server
**Check**:
1. Is `encode_to_base64=True` in client?
2. Does server's `load_image()` handle base64?
3. Check base64 string starts with `data:image` or is valid base64

### Issue: Coordinate precision loss
**Solution**: Use `coordinate_mode="scaled"` for higher precision

## Migration Guide

### From old version to new version:

**Old code:**
```python
# Save image to file
cv2.imwrite("temp_image.jpg", image)

# Pass file path
client.send_manipulation_request(images=["temp_image.jpg"])
```

**New code:**
```python
# Pass numpy array directly (will be base64 encoded automatically)
client.send_manipulation_request(
    images=[image],  # numpy array
    encode_to_base64=True
)
```

No more temporary files needed!


