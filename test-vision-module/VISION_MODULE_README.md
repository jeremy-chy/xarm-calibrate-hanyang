# 视觉模块 (Vision Module) 使用文档

基于 **Grounding DINO** 和 **SAM (Segment Anything Model)** 的目标检测与分割模块

## 📋 功能概述

这个视觉模块提供了一个 `generate_mask` 方法，可以：
- **输入**: 一张 JPG 图片 + 一个文本指令（如"找出所有cube"）
- **输出**: 一个 2D masks 列表

### 工作流程

```
图片 + 指令
    ↓
[Grounding DINO] ─→ 检测目标物体 ─→ 生成 2D Bounding Boxes
    ↓
[SAM] ─→ 根据 Bounding Boxes ─→ 生成精确的 2D Masks
    ↓
返回: List of 2D Masks
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision numpy opencv-python Pillow

# 安装 Grounding DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..

# 安装 SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 或者使用提供的 requirements 文件
pip install -r vision_requirements.txt
```

### 2. 下载模型权重

#### Grounding DINO 模型

```bash
# 创建模型目录
mkdir -p models

# 下载 Grounding DINO 权重文件 (约 662 MB)
wget -P models/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 下载配置文件
# 如果已经克隆了 GroundingDINO 仓库，配置文件在:
# GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

#### SAM 模型

SAM 提供三种规模的模型，选择其中一个：

```bash
# ViT-H (Huge) - 最大最准确 (约 2.4 GB) ⭐ 推荐
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (Large) - 中等 (约 1.2 GB)
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (Base) - 最小最快 (约 375 MB)
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. 基础使用

```python
from vision_module import VisionModule

# 初始化视觉模块
vision = VisionModule(
    grounding_dino_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint_path="models/groundingdino_swint_ogc.pth",
    sam_checkpoint_path="models/sam_vit_h_4b8939.pth",
    sam_model_type="vit_h",
    device="cuda"  # 使用 GPU，如果没有 GPU 使用 "cpu"
)

# 生成 masks
masks, boxes, scores = vision.generate_mask(
    image_path="Xarm_test.jpg",
    instruction="找出所有cube",  # 或 "all cubes", "red cube" 等
    box_threshold=0.35,
    text_threshold=0.25
)

# 使用生成的 masks
for i, mask in enumerate(masks):
    print(f"Mask {i+1} shape: {mask.shape}")  # (H, W)
    print(f"分割像素数: {mask.sum()}")
```

## 📚 详细说明

### VisionModule 类

#### `__init__()` 方法

初始化两个模型：

1. **Grounding DINO**
   - 开放词汇的目标检测模型
   - 根据自然语言描述检测物体
   - 输出: Bounding boxes + 置信度分数

2. **SAM (Segment Anything Model)**
   - 通用图像分割模型
   - 根据提示（bounding box）生成精确 masks
   - 输出: 2D 分割掩码

```python
def __init__(
    self,
    grounding_dino_config_path: str,      # Grounding DINO 配置文件
    grounding_dino_checkpoint_path: str,  # Grounding DINO 权重文件
    sam_checkpoint_path: str,             # SAM 权重文件
    sam_model_type: str = "vit_h",        # SAM 模型类型
    device: str = "cuda"                  # 运行设备
)
```

#### `generate_mask()` 方法

主要功能方法，执行完整的检测和分割流程：

```python
def generate_mask(
    self,
    image_path: str,            # 输入图片路径
    instruction: str,           # 检测指令
    box_threshold: float = 0.35,  # Bounding box 置信度阈值
    text_threshold: float = 0.25  # 文本匹配置信度阈值
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]
```

**返回值:**
- `masks`: List of 2D numpy arrays，每个是一个二值化 mask (H, W)
- `boxes`: List of bounding boxes，格式为 [x1, y1, x2, y2]
- `scores`: List of confidence scores

### 支持的指令格式

Grounding DINO 支持自然语言指令，例如：

- 中文: `"找出所有cube"`, `"红色方块"`, `"绿色的物体"`
- 英文: `"all cubes"`, `"red cube"`, `"green blocks"`
- 组合: `"red cube and green cube"`, `"cube . block"`

## 💡 使用示例

### 示例 1: 检测特定颜色的物体

```python
# 检测红色 cube
masks, boxes, scores = vision.generate_mask(
    image_path="scene.jpg",
    instruction="red cube",
    box_threshold=0.35
)

# 选择置信度最高的
best_idx = np.argmax(scores)
target_mask = masks[best_idx]
```

### 示例 2: 提取 mask 的中心点

```python
import numpy as np

masks, boxes, scores = vision.generate_mask(
    image_path="scene.jpg",
    instruction="找出所有cube"
)

for i, mask in enumerate(masks):
    # 计算 mask 的中心点
    y_coords, x_coords = np.where(mask)
    center_x = x_coords.mean()
    center_y = y_coords.mean()
    print(f"物体 {i+1} 中心: ({center_x:.1f}, {center_y:.1f})")
```

### 示例 3: 可视化结果

```python
# 生成带有 masks 和 boxes 的可视化图片
vision.visualize_results(
    image_path="scene.jpg",
    masks=masks,
    boxes=boxes,
    output_path="result.jpg"
)
```

### 示例 4: 提取被分割的物体

```python
import cv2

image = cv2.imread("scene.jpg")

for i, mask in enumerate(masks):
    # 创建只包含该物体的图片
    masked_image = image.copy()
    masked_image[~mask] = 0  # 将非 mask 区域设为黑色
    cv2.imwrite(f"object_{i+1}.jpg", masked_image)
```

## 🔧 参数调优

### box_threshold (默认: 0.35)

- **含义**: Grounding DINO 检测的置信度阈值
- **范围**: 0.0 ~ 1.0
- **调优建议**:
  - 提高 (0.4 ~ 0.5): 减少误检，只保留高置信度的检测
  - 降低 (0.2 ~ 0.3): 增加检测敏感度，可能检测到更多物体

### text_threshold (默认: 0.25)

- **含义**: 文本匹配的置信度阈值
- **范围**: 0.0 ~ 1.0
- **调优建议**:
  - 提高: 要求更精确的文本匹配
  - 降低: 允许更模糊的匹配

### SAM multimask_output

在 `vision_module.py` 中的 `sam_predictor.predict()` 调用：

```python
# 当前设置: 只输出一个最佳 mask
mask_output, _, _ = self.sam_predictor.predict(
    box=box,
    multimask_output=False  # 改为 True 可以输出 3 个候选 masks
)
```

## 🎯 与机器人系统集成

### 典型工作流程

```python
# 1. 检测和分割目标物体
masks, boxes, scores = vision.generate_mask(
    image_path="camera_frame.jpg",
    instruction="red cube"
)

# 2. 选择目标 (如置信度最高的)
target_idx = np.argmax(scores)
target_mask = masks[target_idx]

# 3. 计算抓取点 (mask 中心)
y_coords, x_coords = np.where(target_mask)
grasp_2d_x = x_coords.mean()
grasp_2d_y = y_coords.mean()

# 4. 2D → 3D 坐标转换 (使用相机内参)
# grasp_3d = pixel_to_world(grasp_2d_x, grasp_2d_y, depth, camera_intrinsics)

# 5. 发送抓取指令给机器人
# robot.move_to(grasp_3d)
```

## 📊 性能优化

### GPU 内存优化

如果遇到 CUDA out of memory 错误：

```python
# 使用较小的 SAM 模型
vision = VisionModule(
    ...,
    sam_checkpoint_path="models/sam_vit_b_01ec64.pth",
    sam_model_type="vit_b",  # 改用 base 版本
    device="cuda"
)
```

### 推理速度优化

```python
# 1. 使用 torch.cuda.amp 进行混合精度推理
# 2. 批量处理多个 bounding boxes
# 3. 图片预处理时调整大小
```

## ⚠️ 常见问题

### 1. 检测不到物体

**解决方案**:
- 降低 `box_threshold` 和 `text_threshold`
- 尝试不同的指令描述
- 检查图片质量和光照条件

### 2. 检测到太多误报

**解决方案**:
- 提高 `box_threshold`
- 使用更具体的指令描述
- 后处理过滤（根据 mask 面积、位置等）

### 3. Mask 不够精确

**解决方案**:
- 使用更大的 SAM 模型 (vit_h)
- 启用 `multimask_output=True` 并选择最佳 mask
- 对 mask 进行后处理（形态学操作）

## 📁 文件结构

```
Xarm/
├── vision_module.py           # 视觉模块主文件
├── vision_example.py          # 使用示例
├── vision_requirements.txt    # 依赖包列表
├── VISION_MODULE_README.md    # 本文档
├── models/                    # 模型权重目录
│   ├── groundingdino_swint_ogc.pth
│   └── sam_vit_h_4b8939.pth
└── GroundingDINO/            # Grounding DINO 源码
    └── groundingdino/config/GroundingDINO_SwinT_OGC.py
```

## 📖 参考资料

- [Grounding DINO 论文](https://arxiv.org/abs/2303.05499)
- [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [SAM 论文](https://arxiv.org/abs/2304.02643)
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)

## 📝 更新日志

- **2025-11-04**: 初始版本，实现基础的检测和分割功能

