# Robot Grasping Module

一个完整的机器人抓取系统，集成了 RealSense 相机、视觉模块（Grounding DINO + SAM）和 xArm6 机械臂。

## 功能概述

这个模块实现了一个完整的抓取流程：

1. **相机初始化** - 配置 RealSense 深度相机和滤波器
2. **拍照** - 获取彩色和深度图像
3. **视觉检测** - 使用 Grounding DINO + SAM 检测和分割目标物体
4. **3D 定位** - 将 2D mask 转换为相机坐标系下的 3D 点
5. **坐标转换** - 将相机坐标系转换到机器人基座坐标系
6. **执行抓取** - 控制机械臂移动并操作夹爪

## 架构

```
robot.py
├── Robot (主类)
│   ├── __init__()              # 初始化相机、机械臂、视觉模块
│   ├── _init_camera()          # 配置 RealSense 相机和滤波器
│   ├── capture_frame()         # 拍照并获取对齐的深度和彩色帧
│   ├── get_object_mask()       # 调用视觉模块获取 2D masks
│   ├── mask_to_camera_3d()     # 将 2D mask 转换为相机系 3D 点
│   ├── camera_to_base_transform() # 转换到基座坐标系
│   ├── grasp()                 # 执行抓取动作（移动+夹爪）
│   ├── execute()               # 主循环执行抓取任务
│   └── cleanup()               # 清理资源
```

## 数据流

```
相机采集
    ↓
[capture_frame]
    ↓
RGB + Depth 对齐帧
    ↓
[get_object_mask]
    ↓
2D Masks (list)
    ↓
[mask_to_camera_3d]
    ↓
3D 点（相机系，米）
    ↓
[camera_to_base_transform]
    ↓
3D 点（基座系，米）
    ↓
转换为毫米
    ↓
[grasp]
    ↓
执行抓取动作
```

## 使用方法

### 1. 基本使用

```python
from robot import Robot

# 初始化机器人
robot = Robot(
    serial_number="your_camera_serial_number",
    grounding_dino_config_path="path/to/config.py",
    grounding_dino_checkpoint_path="path/to/checkpoint.pth",
    sam_checkpoint_path="path/to/sam.pth"
)

# 执行抓取任务
robot.execute(max_steps=10, instruction="find all cubes")

# 清理资源
robot.cleanup()
```

### 2. 完整示例

参见 `example_usage.py`

### 3. 初始化参数说明

```python
Robot(
    serial_number,                      # RealSense 相机序列号
    camera_to_base_path,                # 相机到基座的校准文件路径
    arm_interface,                      # 机械臂 IP 地址
    init_pose,                          # 初始位姿 [X, Y, Z, roll, pitch, yaw]
    init_servo_angle,                   # 初始关节角度
    gripper_enable,                     # 是否启用夹爪
    grounding_dino_config_path,         # Grounding DINO 配置文件
    grounding_dino_checkpoint_path,     # Grounding DINO 权重文件
    sam_checkpoint_path,                # SAM 权重文件
    sam_model_type                      # SAM 模型类型 (vit_h/vit_l/vit_b)
)
```

## 关键实现细节

### 1. `capture_frame()`
基于 `verify_stationary_cameras.py` 的 40-52 行实现
- 获取帧并对齐深度到彩色
- 应用滤波器：disparity → temporal → threshold
- 返回处理后的深度和彩色帧

### 2. `get_object_mask()`
- 调用 `vision_module.py` 的 `generate_mask()`
- 输入：彩色帧、检测指令
- 输出：masks 列表、bounding boxes、置信度分数

### 3. `mask_to_camera_3d()`
基于 `verify_stationary_cameras.py` 的 54-61 行实现
- 为每个 mask：
  1. 生成完整点云
  2. 用 mask 过滤点云
  3. 计算质心作为物体位置
- 输出：相机坐标系下的 3D 点列表（米）

### 4. `camera_to_base_transform()`
基于 `verify_stationary_cameras.py` 的 66-68 行实现
- 使用齐次坐标变换
- 应用校准得到的 4×4 变换矩阵
- 输出：基座坐标系下的 3D 点列表（米）

### 5. `grasp(target_vector)`
**重要：**输入是长度为 4 的向量 `[x, y, z, gripper_state]`
- `(x, y, z)`: 夹爪目标位置（毫米）
- `gripper_state`: 1=打开，0=关闭

**关键点：**
1. **坐标转换**：xArm 的 `move_to_pose` 控制法兰中心位置，需要将夹爪位置的 z 减去 175mm（夹爪高度）
2. **保持方向**：从当前位姿提取 roll, pitch, yaw 并保持不变
3. **智能夹爪控制**：检查当前状态，只在需要时改变夹爪状态

**实现流程：**
```python
# 1. 获取当前方向
current_pose = arm.get_current_pose()
roll, pitch, yaw = current_pose[3:6]

# 2. 计算法兰位置
flange_z = gripper_z - 175.0  # mm

# 3. 移动到目标位置
target_pose = [x, y, flange_z, roll, pitch, yaw]
arm.move_to_pose(target_pose)

# 4. 改变夹爪状态（如果需要）
if target_state != current_state:
    arm.open_gripper() or arm.close_gripper()
```

### 6. `execute()` 抓取流程

对每个检测到的物体执行完整的抓取序列：

```python
# 1. 移动到预抓取位置（物体上方 100mm，夹爪打开）
grasp([x, y, z+100, 1])

# 2. 下降到抓取位置（夹爪仍打开）
grasp([x, y, z, 1])

# 3. 关闭夹爪抓取物体
grasp([x, y, z, 0])

# 4. 提升物体（上升 150mm，夹爪保持关闭）
grasp([x, y, z+150, 0])
```

## 单位说明

- **RealSense 点云**: 米 (m)
- **相机/基座坐标系**: 米 (m)
- **xArm 位置**: 毫米 (mm)
- **转换**: `position_mm = position_m * 1000`

## 坐标系

```
Base Frame (基座坐标系)
    ↑ Z
    |
    |___→ X
   /
  ↙ Y

Camera Frame (相机坐标系)
转换矩阵: camera_to_base_matrix (4×4)
```

## 依赖

- `pyrealsense2` - RealSense 相机 SDK
- `open3d` - 3D 点云处理
- `opencv-python` - 图像处理
- `numpy` - 数值计算
- `torch` - 深度学习框架
- `groundingdino` - 目标检测
- `segment-anything` - 图像分割

## 注意事项

1. **安全性**：确保工作区域内无障碍物
2. **校准**：使用前必须完成相机到基座的校准
3. **单位**：注意米和毫米的转换
4. **夹爪高度**：默认 175mm，如更换夹爪需调整
5. **坐标系**：确保校准文件与当前设置一致

## 故障排除

### 相机初始化失败
- 检查相机序列号是否正确
- 确认相机已正确连接
- 检查是否有其他程序占用相机

### 检测不到物体
- 调整 `box_threshold` 和 `text_threshold`
- 修改检测指令使其更具体
- 检查光照条件

### 抓取位置不准确
- 重新校准相机到基座的变换
- 检查深度数据质量
- 调整滤波器参数

### 机械臂移动异常
- 检查目标位置是否在工作空间内
- 确认机械臂连接正常
- 检查是否有碰撞或限位

## 开发者信息

基于以下文件开发：
- `verify_stationary_cameras.py` - 相机处理逻辑
- `vision_module.py` - 视觉检测模块
- `xarm6.py` - 机械臂控制接口
- `real_env.py` - 环境设置参考


