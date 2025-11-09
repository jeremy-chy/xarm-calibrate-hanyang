"""
简化的视觉模块使用示例
使用配置文件，更容易上手
"""

from vision_module import VisionModule
from vision_config import get_vision_config, verify_config
import numpy as np


def main():
    """
    简单的使用演示
    """
    print("=" * 70)
    print("视觉模块演示 - 检测和分割物体")
    print("=" * 70)
    print()
    
    # ========== 步骤 1: 验证配置 ==========
    print("步骤 1: 验证配置...")
    if not verify_config():
        print("\n请先运行 setup_vision_models.sh 下载模型")
        return
    print()
    
    # ========== 步骤 2: 初始化视觉模块 ==========
    print("步骤 2: 初始化视觉模块...")
    try:
        # 使用配置文件中的设置
        config = get_vision_config()
        vision = VisionModule(**config)
        print("✓ 视觉模块初始化成功")
        print()
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        print("\n请检查:")
        print("1. 是否已安装所有依赖: pip install -r vision_requirements.txt")
        print("2. 是否已安装 Grounding DINO: cd GroundingDINO && pip install -e .")
        print("3. 是否已安装 SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return
    
    # ========== 步骤 3: 选择测试图片 ==========
    print("步骤 3: 选择测试图片...")
    
    # 使用当前目录中的图片
    import os
    test_images = [
        "Xarm_test.jpg",
        "Xarm_test_resized.jpg",
        "2.jpg"
    ]
    
    # 找到第一个存在的图片
    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if image_path is None:
        print("✗ 未找到测试图片")
        print(f"请在当前目录放置以下图片之一: {test_images}")
        return
    
    print(f"✓ 使用图片: {image_path}")
    print()
    
    # ========== 步骤 4: 执行检测和分割 ==========
    print("步骤 4: 执行检测和分割...")
    print()
    
    # 定义要检测的指令
    instructions = [
        "get all cubes"
    ]
    
    for instruction in instructions:
        print("-" * 70)
        print(f"检测指令: {instruction}")
        print("-" * 70)
        
        try:
            # 调用 generate_mask 方法
            masks, boxes, scores = vision.generate_mask(
                image_path=image_path,
                instruction=instruction,
                box_threshold=0.35,
                text_threshold=0.25
            )

            print(masks)
            print(type(masks))
            print(masks[0])
            print(type(masks[0]))
            
            # ========== 步骤 5: 处理结果 ==========
            if len(masks) == 0:
                print("未检测到任何物体")
                print("提示: 尝试降低 box_threshold 或使用不同的指令")
            else:
                print(f"\n✓ 成功检测到 {len(masks)} 个物体")
                print()
                
                # 显示每个检测到的物体的详细信息
                for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                    print(f"物体 {i+1}:")
                    print(f"  - 置信度: {score:.3f}")
                    print(f"  - Bounding Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                    print(f"  - Mask 尺寸: {mask.shape}")
                    print(f"  - 分割像素数: {mask.sum()}")
                    
                    # 计算中心点
                    y_coords, x_coords = np.where(mask)
                    if len(x_coords) > 0:
                        center_x = x_coords.mean()
                        center_y = y_coords.mean()
                        print(f"  - 中心点坐标: ({center_x:.1f}, {center_y:.1f})")
                    print()
                
                # ========== 步骤 6: 可视化结果 ==========
                output_path = f"result_{instruction.replace(' ', '_')}.jpg"
                vision.visualize_results(
                    image_path=image_path,
                    masks=masks,
                    boxes=boxes,
                    output_path=output_path
                )
                print(f"✓ 可视化结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"✗ 检测失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 70)
    print("演示完成！")
    print("=" * 70)
    print()
    print("接下来你可以:")
    print("1. 尝试不同的检测指令（修改 instructions 列表）")
    print("2. 调整检测阈值（box_threshold 和 text_threshold）")
    print("3. 将生成的 masks 用于机器人控制")
    print()


if __name__ == "__main__":
    """
    运行简化演示
    
    使用前确保:
    1. 已运行 setup_vision_models.sh 下载模型
    2. 已安装所有依赖包
    3. 当前目录有测试图片
    """
    main()

