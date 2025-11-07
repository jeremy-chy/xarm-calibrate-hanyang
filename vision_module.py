"""
视觉模块 (Vision Module)
功能: 使用 Grounding DINO 和 SAM 从图片中生成指定物体的 masks
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import cv2

# Grounding DINO imports
from groundingdino.util.inference import load_model as load_grounding_dino_model
from groundingdino.util.inference import predict as grounding_dino_predict
from groundingdino.util.inference import annotate as grounding_dino_annotate
import groundingdino.datasets.transforms as T

# SAM (Segment Anything Model) imports
from segment_anything import sam_model_registry, SamPredictor


class VisionModule:
    """
    视觉模块类
    
    该模块结合了 Grounding DINO 和 SAM 两个模型:
    1. Grounding DINO: 根据文本指令检测目标物体，生成 2D bounding boxes
    2. SAM: 根据 bounding boxes 生成精确的分割 masks
    """
    
    def __init__(
        self,
        grounding_dino_config_path: str,
        grounding_dino_checkpoint_path: str,
        sam_checkpoint_path: str,
        sam_model_type: str = "vit_h",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化视觉模块
        
        Args:
            grounding_dino_config_path: Grounding DINO 配置文件路径
            grounding_dino_checkpoint_path: Grounding DINO 模型权重路径
            sam_checkpoint_path: SAM 模型权重路径
            sam_model_type: SAM 模型类型 (vit_h, vit_l, vit_b)
            device: 运行设备 (cuda/cpu)
        """
        self.device = device
        print(f"使用设备: {self.device}")
        
        # ========== 初始化 Grounding DINO 模型 ==========
        # Grounding DINO 是一个开放词汇的目标检测模型
        # 它可以根据自然语言描述（如"找出所有cube"）来检测图片中的物体
        # 输出: bounding boxes (边界框) 和对应的置信度分数
        print("正在加载 Grounding DINO 模型...")
        self.grounding_dino_model = load_grounding_dino_model(
            grounding_dino_config_path,
            grounding_dino_checkpoint_path,
            device=self.device
        )
        print("✓ Grounding DINO 模型加载完成")
        
        # ========== 初始化 SAM (Segment Anything Model) ==========
        # SAM 是一个通用的图像分割模型
        # 它可以根据提示（如 bounding box, point, mask）生成精确的分割掩码
        # 在这里，我们使用 Grounding DINO 检测到的 bounding boxes 作为提示
        print("正在加载 SAM 模型...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        
        # SamPredictor 是 SAM 的预测器，用于对单张图片进行推理
        self.sam_predictor = SamPredictor(sam)
        print("✓ SAM 模型加载完成")
        
        # Grounding DINO 的图像预处理 transform
        self.grounding_dino_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def generate_mask(
        self,
        image_path: str,
        instruction: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        生成 masks 的主函数
        
        Pipeline:
        1. 加载图片
        2. 使用 Grounding DINO 根据指令检测目标物体 → 得到 bounding boxes
        3. 使用 SAM 根据 bounding boxes 生成分割 masks
        
        Args:
            image_path: 输入图片路径 (.jpg)
            instruction: 检测指令 (例如: "找出所有cube", "all red blocks")
            box_threshold: Grounding DINO 的 box 置信度阈值
            text_threshold: Grounding DINO 的 text 置信度阈值
        
        Returns:
            masks: List of 2D numpy arrays, 每个 mask 是一个二值化的分割掩码
            boxes: List of bounding boxes (xyxy format)
            scores: List of confidence scores
        """
        
        # ========== Step 1: 加载和预处理图片 ==========
        print(f"\n开始处理图片: {image_path}")
        print(f"检测指令: {instruction}")
        
        # 使用 PIL 加载图片 (Grounding DINO 需要)
        image_pil = Image.open(image_path).convert("RGB")
        
        # 使用 OpenCV 加载图片 (SAM 需要 BGR 格式)
        image_cv2 = cv2.imread(image_path)
        image_cv2_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        
        # ========== Step 2: 使用 Grounding DINO 检测目标物体 ==========
        print("\n[Grounding DINO] 开始检测目标物体...")
        
        # 对图片进行预处理 transform
        image_transformed, _ = self.grounding_dino_transform(image_pil, None)
        
        # 使用 Grounding DINO 进行推理
        # 输入: 图片 + 文本指令
        # 输出: bounding boxes, 置信度分数, 文本标签
        boxes, logits, phrases = grounding_dino_predict(
            model=self.grounding_dino_model,
            image=image_transformed,
            caption=instruction,  # 文本指令
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Grounding DINO 输出的 boxes 是归一化的 (0-1 范围) 且格式为 cxcywh (中心点坐标+宽高)
        # 需要转换为实际像素坐标，并从 cxcywh 转换为 xyxy 格式
        h, w = image_cv2_rgb.shape[:2]
        
        # 先转换为像素坐标 (仍然是 cxcywh 格式)
        boxes_cxcywh = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        
        # 从 cxcywh 转换为 xyxy 格式
        # xyxy[0] = cx - w/2  (x1)
        # xyxy[1] = cy - h/2  (y1)
        # xyxy[2] = cx + w/2  (x2)
        # xyxy[3] = cy + h/2  (y2)
        boxes_xyxy = torch.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2
        
        boxes_xyxy = boxes_xyxy.cpu().numpy()
        
        print(f"[Grounding DINO] 检测到 {len(boxes_xyxy)} 个目标物体")
        for i, (box, score, phrase) in enumerate(zip(boxes_xyxy, logits, phrases)):
            print(f"  目标 {i+1}: {phrase} (置信度: {score:.3f})")
            print(f"         边界框 (xyxy): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        
        # ========== 可视化 Grounding DINO 检测结果 ==========
        # 在SAM分割之前，先可视化检测到的边界框，用于调试
        self._visualize_detection_boxes(
            image_cv2_rgb, 
            boxes_xyxy, 
            logits.cpu().numpy(), 
            phrases,
            save_path="debug_grounding_dino_detection.jpg"
        )
        print("[调试] Grounding DINO 检测结果已保存到: debug_grounding_dino_detection.jpg")
        
        # ========== Step 3: 使用 SAM 生成分割 masks ==========
        print("\n[SAM] 开始生成分割 masks...")
        
        # 设置图片到 SAM predictor
        # SAM 会对图片进行特征提取，这个过程只需要做一次
        self.sam_predictor.set_image(image_cv2_rgb)
        
        masks = []
        
        # 遍历每个检测到的 bounding box
        for i, box in enumerate(boxes_xyxy):
            # 使用 bounding box 作为提示（prompt）来生成 mask
            # SAM 可以接受多种提示：box, point, mask
            # 这里我们使用 Grounding DINO 检测到的 box 作为提示
            
            # SAM 的 predict 方法:
            # - box: bounding box 坐标 (xyxy format)
            # - multimask_output=False: 只输出一个最佳 mask
            #   如果设为 True，SAM 会输出 3 个候选 masks
            mask_output, scores_sam, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=False
            )
            
            # mask_output shape: (1, H, W), 二值化 mask
            mask = mask_output[0]  # 取第一个 mask
            masks.append(mask)
            
            print(f"  目标 {i+1}: 生成 mask (尺寸: {mask.shape}, 分割像素数: {mask.sum()})")
        
        print(f"\n[完成] 成功生成 {len(masks)} 个 masks")
        
        return masks, boxes_xyxy, logits.cpu().numpy()
    
    def _visualize_detection_boxes(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: List[str],
        save_path: str
    ):
        """
        可视化 Grounding DINO 检测到的边界框（用于调试）
        
        Args:
            image_rgb: RGB格式的图片
            boxes: bounding boxes 数组 (N, 4) xyxy format
            scores: 置信度分数数组 (N,)
            labels: 标签列表
            save_path: 保存路径
        """
        # 创建图片副本
        result_image = image_rgb.copy()
        
        # 绘制每个边界框
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box.astype(int)
            
            # 绘制边界框 (绿色)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签和置信度
            text = f"{i+1}: {label} ({score:.2f})"
            cv2.putText(result_image, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
    
    def visualize_results(
        self,
        image_path: str,
        masks: List[np.ndarray],
        boxes: List[np.ndarray],
        output_path: str = None
    ):
        """
        可视化检测和分割结果
        
        Args:
            image_path: 原始图片路径
            masks: 生成的 masks 列表
            boxes: bounding boxes 列表
            output_path: 输出图片保存路径 (如果为 None，则不保存)
        """
        # 加载原始图片
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建一个副本用于绘制
        result_image = image_rgb.copy()
        
        # 为每个 mask 使用不同的颜色
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 青色
        ]
        
        # 绘制 masks
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            # 创建彩色 mask overlay
            colored_mask = np.zeros_like(result_image)
            colored_mask[mask] = color
            # 半透明叠加
            result_image = cv2.addWeighted(result_image, 1.0, colored_mask, 0.5, 0)
        
        # 绘制 bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[i % len(colors)]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            # 添加标签
            cv2.putText(result_image, f"Object {i+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存或显示结果
        if output_path:
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            print(f"可视化结果已保存到: {output_path}")
        
        return result_image


# ========== 示例使用代码 ==========
if __name__ == "__main__":
    """
    使用示例
    
    注意: 需要先下载 Grounding DINO 和 SAM 的模型权重文件
    """
    
    # 初始化视觉模块
    vision = VisionModule(
        grounding_dino_config_path="path/to/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint_path="path/to/groundingdino_swint_ogc.pth",
        sam_checkpoint_path="path/to/sam_vit_h_4b8939.pth",
        sam_model_type="vit_h",
        device="cuda"
    )
    
    # 生成 masks
    masks, boxes, scores = vision.generate_mask(
        image_path="example.jpg",
        instruction="find all cubes",  # 或 "all cubes"
        box_threshold=0.35,
        text_threshold=0.25
    )

    print(masks)
    print(type(masks))
    print(masks[0])
    print(type(masks[0]))
    
    # 可视化结果
    vision.visualize_results(
        image_path="example.jpg",
        masks=masks,
        boxes=boxes,
        output_path="result_with_masks.jpg"
    )
    
    # 使用生成的 masks
    for i, mask in enumerate(masks):
        print(f"Mask {i+1} shape: {mask.shape}")
        print(f"Mask {i+1} type: {mask.dtype}")
        # mask 是一个二值化的 numpy array (True/False 或 1/0)
        # 可以用于后续的图像处理、坐标提取等操作

