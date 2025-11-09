"""
视觉模块配置文件
集中管理所有模型路径和参数
"""

import os

# ========== 项目路径配置 ==========
# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== Grounding DINO 配置 ==========
GROUNDING_DINO_CONFIG = os.path.join(
    BASE_DIR,
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

GROUNDING_DINO_CHECKPOINT = os.path.join(
    BASE_DIR,
    "models/groundingdino_swint_ogc.pth"
)

# ========== SAM 配置 ==========
# SAM 模型类型选择: "vit_h", "vit_l", "vit_b"
SAM_MODEL_TYPE = "vit_h"

# 根据模型类型自动选择对应的权重文件
SAM_CHECKPOINTS = {
    "vit_h": os.path.join(BASE_DIR, "models/sam_vit_h_4b8939.pth"),
    "vit_l": os.path.join(BASE_DIR, "models/sam_vit_l_0b3195.pth"),
    "vit_b": os.path.join(BASE_DIR, "models/sam_vit_b_01ec64.pth"),
}

SAM_CHECKPOINT = SAM_CHECKPOINTS.get(SAM_MODEL_TYPE)

# ========== 设备配置 ==========
# 自动检测 CUDA 是否可用
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 检测参数配置 ==========
# Grounding DINO 的检测阈值
DEFAULT_BOX_THRESHOLD = 0.35  # Bounding box 置信度阈值
DEFAULT_TEXT_THRESHOLD = 0.25  # 文本匹配置信度阈值

# ========== 验证配置 ==========
def verify_config():
    """
    验证配置是否正确
    检查所有必需的文件是否存在
    """
    errors = []
    
    # 检查 Grounding DINO 配置文件
    if not os.path.exists(GROUNDING_DINO_CONFIG):
        errors.append(f"Grounding DINO 配置文件不存在: {GROUNDING_DINO_CONFIG}")
    
    # 检查 Grounding DINO 权重文件
    if not os.path.exists(GROUNDING_DINO_CHECKPOINT):
        errors.append(f"Grounding DINO 权重文件不存在: {GROUNDING_DINO_CHECKPOINT}")
    
    # 检查 SAM 权重文件
    if not os.path.exists(SAM_CHECKPOINT):
        errors.append(f"SAM 权重文件不存在: {SAM_CHECKPOINT}")
    
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        print("\n请运行以下命令下载模型:")
        print("  bash setup_vision_models.sh")
        return False
    else:
        print("✅ 配置验证成功！")
        print(f"  - Grounding DINO 配置: {os.path.basename(GROUNDING_DINO_CONFIG)}")
        print(f"  - Grounding DINO 权重: {os.path.basename(GROUNDING_DINO_CHECKPOINT)}")
        print(f"  - SAM 模型类型: {SAM_MODEL_TYPE}")
        print(f"  - SAM 权重: {os.path.basename(SAM_CHECKPOINT)}")
        print(f"  - 运行设备: {DEVICE}")
        return True


def get_vision_config():
    """
    获取视觉模块配置字典
    可以直接传递给 VisionModule 构造函数
    """
    return {
        "grounding_dino_config_path": GROUNDING_DINO_CONFIG,
        "grounding_dino_checkpoint_path": GROUNDING_DINO_CHECKPOINT,
        "sam_checkpoint_path": SAM_CHECKPOINT,
        "sam_model_type": SAM_MODEL_TYPE,
        "device": DEVICE,
    }


# ========== 使用示例 ==========
if __name__ == "__main__":
    """
    运行此脚本来验证配置
    """
    print("=" * 60)
    print("视觉模块配置验证")
    print("=" * 60)
    print()
    
    # 显示配置信息
    print("当前配置:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  DEVICE: {DEVICE}")
    print(f"  SAM_MODEL_TYPE: {SAM_MODEL_TYPE}")
    print()
    
    # 验证配置
    verify_config()
    print()
    
    # 显示配置字典
    print("=" * 60)
    print("VisionModule 配置字典:")
    print("=" * 60)
    config = get_vision_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

