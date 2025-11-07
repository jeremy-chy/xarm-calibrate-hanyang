# Policy-Based Robot Control Architecture

## 概述

Robot 类现在采用了 **policy-based** 的控制架构，将决策逻辑（policy）与执行逻辑（grasp）分离。

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    Execute Loop                         │
│                                                         │
│  Step 1-4: Perception                                   │
│    ├── capture_frame()                                  │
│    ├── get_object_mask()                                │
│    ├── mask_to_camera_3d()                              │
│    └── camera_to_base_transform()                       │
│         │                                               │
│         ↓                                               │
│  Step 5: Policy & Action                                │
│    ├── policy(step, color_image, points_base)          │
│    │     → returns 4D action [x, y, z, gripper]        │
│    │                                                     │
│    └── grasp(action)                                    │
│          → executes the action                          │
└─────────────────────────────────────────────────────────┘
```

## Policy 函数

### 函数签名

```python
def policy(self, step, color_image, points_base):
    """
    Determine the next action based on current state.
    
    Args:
        step: Current step number (1-indexed)
        color_image: Color image from camera (numpy array, RGB format)
        points_base: List of target object positions in base frame (meters)
    
    Returns:
        action: 4D vector [x, y, z, gripper_state] in millimeters
    """
```

### 输入参数详解

1. **`step`**: 当前步骤编号（从 1 开始）
   - 可用于实现分阶段的控制策略
   - 例如：step 1 抓取，step 2 提升，step 3+ 保持

2. **`color_image`**: 当前帧的彩色图像
   - 格式：numpy array (H, W, 3), RGB
   - 可用于基于视觉的决策
   - 未来可用于更高级的策略（如视觉伺服）

3. **`points_base`**: 检测到的物体位置列表
   - 格式：List of numpy arrays, 每个是 [x, y, z] 米
   - 基座坐标系
   - 按检测置信度或其他标准排序

### 输出格式

4D 向量 `[x, y, z, gripper_state]`:
- **x, y, z**: 夹爪目标位置（毫米）
- **gripper_state**: 0 = 关闭，1 = 打开

## 当前实现的 Policy

### 简单抓取-提升 Policy

```python
if step == 1:
    # 抓取第一个物体
    target = points_base[0] * 1000  # 米 → 毫米
    action = [target[0], target[1], target[2], 0]  # 夹爪关闭
    
elif step == 2:
    # 提升物体 20cm
    action = [last_x, last_y, last_z + 200, 0]  # 保持夹爪关闭
    
else:
    # 保持位置
    action = [last_x, last_y, last_z + 200, 0]
```

### 执行流程示例

```
Step 1:
  Perception → 检测到 3 个物体 at [x1,y1,z1], [x2,y2,z2], [x3,y3,z3]
  Policy(1)  → 返回 [x1,y1,z1,0] (抓取第一个物体)
  Grasp      → 移动到 [x1,y1,z1]，关闭夹爪

Step 2:
  Perception → (同样的物体)
  Policy(2)  → 返回 [x1,y1,z1+200,0] (提升 20cm)
  Grasp      → 移动到 [x1,y1,z1+200]，保持夹爪关闭

Step 3:
  Perception → (同样的物体)
  Policy(3)  → 返回 [x1,y1,z1+200,0] (保持位置)
  Grasp      → 保持位置和夹爪状态
```

## Execute 函数的改动

### 修改前（复杂的 for loop）

```python
# Step 5: Execute grasp for each detected object
for i, target_position in enumerate(points_base):
    # 1. Pre-grasp
    self.grasp([x, y, z+100, 1])
    # 2. Approach
    self.grasp([x, y, z, 1])
    # 3. Grasp
    self.grasp([x, y, z, 0])
    # 4. Lift
    self.grasp([x, y, z+150, 0])
```

### 修改后（简洁的 policy-based）

```python
# Step 5: Execute action from policy
color_image = np.asanyarray(color_frame.get_data())
action = self.policy(step + 1, color_image, points_base)
self.grasp(action)
```

## 优势

### 1. **分离关注点**
- **Policy**: 决策逻辑（"做什么"）
- **Grasp**: 执行逻辑（"怎么做"）

### 2. **灵活性**
- 可以轻松替换不同的 policy
- Policy 可以基于视觉、历史状态、学习等

### 3. **可扩展性**
- 未来可以使用学习的 policy
- 可以添加更复杂的状态管理
- 可以实现条件分支逻辑

### 4. **简洁性**
- Execute loop 更简单、更清晰
- 每一步只做一件事

## 扩展 Policy 的示例

### 示例 1: 基于视觉的 Policy

```python
def policy(self, step, color_image, points_base):
    # 使用颜色信息选择目标
    if step == 1:
        # 分析图像，选择红色物体
        red_mask = self.detect_red_objects(color_image)
        target_idx = self.find_closest_to_mask(red_mask, points_base)
        target = points_base[target_idx] * 1000
        return [target[0], target[1], target[2], 0]
```

### 示例 2: 序列化抓取多个物体

```python
def policy(self, step, color_image, points_base):
    num_objects = len(points_base)
    
    # 每个物体 4 个步骤：pre-grasp, approach, grasp, lift
    obj_idx = (step - 1) // 4
    sub_step = (step - 1) % 4
    
    if obj_idx >= num_objects:
        return self.home_position  # 完成所有物体，返回初始位置
    
    target = points_base[obj_idx] * 1000
    
    if sub_step == 0:    # Pre-grasp
        return [target[0], target[1], target[2]+100, 1]
    elif sub_step == 1:  # Approach
        return [target[0], target[1], target[2], 1]
    elif sub_step == 2:  # Grasp
        return [target[0], target[1], target[2], 0]
    else:                # Lift
        return [target[0], target[1], target[2]+150, 0]
```

### 示例 3: 学习的 Policy（未来）

```python
def policy(self, step, color_image, points_base):
    # 使用神经网络
    state = {
        'image': color_image,
        'objects': points_base,
        'step': step,
        'history': self.action_history
    }
    action = self.policy_network(state)
    return action
```

## 状态管理

Policy 可以维护内部状态：

```python
def __init__(self, ...):
    ...
    self.last_grasp_position = None  # 上次抓取的位置
    self.action_history = []         # 动作历史
    self.object_grasped = False      # 是否已抓取物体
```

在 policy 中使用：

```python
def policy(self, step, color_image, points_base):
    if not self.object_grasped:
        # 还没抓取，执行抓取
        self.object_grasped = True
        ...
    else:
        # 已经抓取，执行其他动作
        ...
    
    # 记录动作历史
    self.action_history.append(action)
```

## 总结

新的 policy-based 架构：
- ✅ **更清晰**: 决策与执行分离
- ✅ **更灵活**: 易于修改策略
- ✅ **更强大**: 支持复杂的多步骤任务
- ✅ **更易测试**: 可以独立测试 policy 和 grasp
- ✅ **面向未来**: 为学习型策略做好准备

---

**修改日期**: 2025-11-06
**架构版本**: v2.0 (Policy-Based)


