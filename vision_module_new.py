"""
视觉模块 (Vision Module)
功能: 使用 Qwen VLM 和 SAM 从图片中生成指定物体的 masks
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import cv2
import base64
import json
import os
from openai import OpenAI

# SAM (Segment Anything Model) imports
from segment_anything import sam_model_registry, SamPredictor


class VisionModule:
    """
    视觉模块类
    
    该模块结合了 Qwen VLM 和 SAM 两个模型:
    1. Qwen VLM: 根据文本指令检测目标物体，生成 2D bounding boxes
    2. SAM: 根据 bounding boxes 生成精确的分割 masks
    """
    
    def __init__(
        self,
        sam_checkpoint_path: str,
        sam_model_type: str = "vit_h",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        api_key: str = None,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    ):
        """
        初始化视觉模块
        
        Args:
            sam_checkpoint_path: SAM 模型权重路径
            sam_model_type: SAM 模型类型 (vit_h, vit_l, vit_b)
            device: 运行设备 (cuda/cpu)
            api_key: Qwen API Key (如果为 None，则从环境变量 DASHSCOPE_API_KEY 读取)
            base_url: API 基础 URL
        """
        self.device = device
        print(f"使用设备: {self.device}")
        
        # ========== 初始化 Qwen VLM 客户端 ==========
        # Qwen VLM 是一个视觉语言模型，可以根据自然语言描述检测图片中的物体
        # 通过 OpenAI 兼容的 API 接口调用
        print("正在初始化 Qwen VLM 客户端...")
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url
        )
        self.model_name = "qwen3-vl-32b-instruct"
        print("✓ Qwen VLM 客户端初始化完成")
        
        # ========== 初始化 SAM (Segment Anything Model) ==========
        # SAM 是一个通用的图像分割模型
        # 它可以根据提示（如 bounding box, point, mask）生成精确的分割掩码
        # 在这里，我们使用 Qwen VLM 检测到的 bounding boxes 作为提示
        print("正在加载 SAM 模型...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        
        # SamPredictor 是 SAM 的预测器，用于对单张图片进行推理
        self.sam_predictor = SamPredictor(sam)
        print("✓ SAM 模型加载完成")
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为 base64 字符串"""
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    def _detect_objects_with_qwen(self, image_path: str, instruction: str) -> dict:
        """
        使用 Qwen VLM 检测图片中的物体
        
        Args:
            image_path: 图片路径
            instruction: 检测指令
        
        Returns:
            包含 bounding boxes 的字典，格式: {"cubes": [{"id": 0, "bbox": {...}}]}
        """
        # 将图片编码为 base64
        b64 = self._encode_image_to_base64(image_path)
        
        # 构造提示词
        user_prompt = (
            f"{instruction}\n"
            "For each object, return its bounding box coordinates.\n"
            "Follow the format below:\n"
            "{\n"
            '  "objects": [\n'
            "    {\n"
            '      "id": 0,\n'
            '      "bbox": {\n'
            '        "x_min": <int>,\n'
            '        "y_min": <int>,\n'
            '        "x_max": <int>,\n'
            '        "y_max": <int>\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Coordinates should be in the range [0, 1000].\n"
            "If there are no objects, return {\"objects\": []}."
        )
        
        # 调用 Qwen VLM API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        }
                    ]
                }
            ],
            temperature=0.0
        )
        
        # 解析返回结果
        content = response.choices[0].message.content.strip()
        
        # 去除可能的 markdown 代码块标记
        if content.startswith("```json"):
            content = content[len("```json"):].strip()
        if content.startswith("```"):
            content = content[len("```"):].strip()
        if content.endswith("```"):
            content = content[:-len("```")].strip()
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[错误] Qwen VLM 返回的内容不是合法 JSON: {content}")
            return None
        
        return result
    
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
        2. 使用 Qwen VLM 根据指令检测目标物体 → 得到 bounding boxes
        3. 使用 SAM 根据 bounding boxes 生成分割 masks
        
        Args:
            image_path: 输入图片路径 (.jpg)
            instruction: 检测指令 (例如: "找出所有cube", "Find all three cubes in the image")
            box_threshold: (保留参数以兼容旧接口，Qwen VLM 中未使用)
            text_threshold: (保留参数以兼容旧接口，Qwen VLM 中未使用)
        
        Returns:
            masks: List of 2D numpy arrays, 每个 mask 是一个二值化的分割掩码
            boxes: List of bounding boxes (xyxy format, 像素坐标)
            scores: List of confidence scores (Qwen VLM 不返回 scores，设为 1.0)
        """
        
        # ========== Step 1: 加载和预处理图片 ==========
        print(f"\n开始处理图片: {image_path}")
        print(f"检测指令: {instruction}")
        
        # 使用 OpenCV 加载图片 (SAM 需要 RGB 格式)
        image_cv2 = cv2.imread(image_path)
        image_cv2_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        h, w = image_cv2_rgb.shape[:2]
        
        # ========== Step 2: 使用 Qwen VLM 检测目标物体 ==========
        print("\n[Qwen VLM] 开始检测目标物体...")
        
        # 调用 Qwen VLM API
        detection_result = self._detect_objects_with_qwen(image_path, instruction)
        
        if detection_result is None:
            print("[警告] 检测失败 (JSON解析错误)")
            return None, None, None

        # 解析检测结果
        objects = detection_result.get("objects", [])
        print(f"[Qwen VLM] 检测到 {len(objects)} 个目标物体")
        
        # 转换坐标：从相对坐标 (0-1000) 转换为像素坐标 (xyxy 格式)
        boxes_xyxy = []
        labels = []
        
        for i, object in enumerate(objects):
            object_id = object.get("id", i)
            bbox = object.get("bbox", {})
            
            # 提取相对坐标 (0-1000)
            rel_x_min = bbox.get("x_min")
            rel_y_min = bbox.get("y_min")
            rel_x_max = bbox.get("x_max")
            rel_y_max = bbox.get("y_max")
            
            if rel_x_min is None or rel_y_min is None or rel_x_max is None or rel_y_max is None:
                print(f"  [警告] 目标 {i+1} 的 bbox 数据不完整，跳过")
                continue
            
            # 转换为像素坐标
            abs_x_min = int(rel_x_min / 1000 * w)
            abs_y_min = int(rel_y_min / 1000 * h)
            abs_x_max = int(rel_x_max / 1000 * w)
            abs_y_max = int(rel_y_max / 1000 * h)
            
            boxes_xyxy.append([abs_x_min, abs_y_min, abs_x_max, abs_y_max])
            labels.append(f"Object {object_id}")
            
            print(f"  目标 {i+1}: Object {object_id}")
            print(f"         边界框 (xyxy): [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")
        
        boxes_xyxy = np.array(boxes_xyxy)
        
        # 如果没有检测到任何物体，返回空列表
        if len(boxes_xyxy) == 0:
            print("[Qwen VLM] 未检测到任何目标物体")
            return [], [], []
        
        # Qwen VLM 不返回置信度分数，统一设为 1.0
        scores = np.ones(len(boxes_xyxy))
        
        # ========== 可视化 Qwen VLM 检测结果 ==========
        # 在SAM分割之前，先可视化检测到的边界框，用于调试
        self._visualize_detection_boxes(
            image_cv2_rgb, 
            boxes_xyxy, 
            scores, 
            labels,
            save_path="debug_qwen_detection.jpg"
        )
        print("[调试] Qwen VLM 检测结果已保存到: debug_qwen_detection.jpg")
        
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
        
        return masks, boxes_xyxy, scores
    
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
    
    注意: 
    1. 需要先下载 SAM 的模型权重文件
    2. 需要设置环境变量 DASHSCOPE_API_KEY (Qwen API Key)
    """
    
    # 初始化视觉模块
    vision = VisionModule(
        sam_checkpoint_path="path/to/sam_vit_h_4b8939.pth",
        sam_model_type="vit_h",
        device="cuda"
        # api_key 会自动从环境变量 DASHSCOPE_API_KEY 读取
    )
    
    # 生成 masks
    masks, boxes, scores = vision.generate_mask(
        image_path="example.jpg",
        instruction="Find all three cubes in the image",  # 检测指令
    )

    print(f"检测到 {len(masks)} 个物体")
    print(f"Masks 类型: {type(masks)}")
    if len(masks) > 0:
        print(f"第一个 Mask 类型: {type(masks[0])}")
        print(f"第一个 Mask 形状: {masks[0].shape}")
    
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

