#!/bin/bash

# 视觉模块模型下载脚本
# 自动下载 Grounding DINO 和 SAM 的模型权重

echo "=================================="
echo "视觉模块设置脚本"
echo "=================================="
echo ""

# 创建模型目录
echo "1. 创建模型目录..."
mkdir -p models
mkdir -p GroundingDINO
cd models
echo "✓ 模型目录已创建: $(pwd)"
echo ""

# 下载 Grounding DINO 模型
echo "2. 下载 Grounding DINO 模型..."
if [ -f "groundingdino_swint_ogc.pth" ]; then
    echo "✓ Grounding DINO 模型已存在，跳过下载"
else
    echo "正在下载 groundingdino_swint_ogc.pth (约 662 MB)..."
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    
    if [ $? -eq 0 ]; then
        echo "✓ Grounding DINO 模型下载完成"
    else
        echo "✗ Grounding DINO 模型下载失败"
        echo "请手动下载: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    fi
fi
echo ""

# 下载 SAM 模型
echo "3. 下载 SAM 模型..."
echo "请选择 SAM 模型大小:"
echo "  1) ViT-H (Huge) - 最准确 (约 2.4 GB) [推荐]"
echo "  2) ViT-L (Large) - 中等 (约 1.2 GB)"
echo "  3) ViT-B (Base) - 最快 (约 375 MB)"
echo "  4) 跳过 SAM 下载"
echo ""
read -p "请输入选项 (1-4): " sam_choice

case $sam_choice in
    1)
        SAM_FILE="sam_vit_h_4b8939.pth"
        SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        ;;
    2)
        SAM_FILE="sam_vit_l_0b3195.pth"
        SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        ;;
    3)
        SAM_FILE="sam_vit_b_01ec64.pth"
        SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        ;;
    4)
        echo "跳过 SAM 模型下载"
        SAM_FILE=""
        ;;
    *)
        echo "无效选项，默认下载 ViT-H"
        SAM_FILE="sam_vit_h_4b8939.pth"
        SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        ;;
esac

if [ -n "$SAM_FILE" ]; then
    if [ -f "$SAM_FILE" ]; then
        echo "✓ SAM 模型已存在，跳过下载"
    else
        echo "正在下载 $SAM_FILE ..."
        wget "$SAM_URL"
        
        if [ $? -eq 0 ]; then
            echo "✓ SAM 模型下载完成"
        else
            echo "✗ SAM 模型下载失败"
            echo "请手动下载: $SAM_URL"
        fi
    fi
fi
echo ""

# 返回上级目录
cd ..

# 克隆 Grounding DINO 仓库（如果需要）
echo "4. 检查 Grounding DINO 源码..."
if [ -d "GroundingDINO/.git" ]; then
    echo "✓ Grounding DINO 仓库已存在"
else
    echo "正在克隆 Grounding DINO 仓库..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    
    if [ $? -eq 0 ]; then
        echo "✓ Grounding DINO 仓库克隆完成"
    else
        echo "✗ Grounding DINO 仓库克隆失败"
        echo "请手动克隆: git clone https://github.com/IDEA-Research/GroundingDINO.git"
    fi
fi
echo ""

# 显示已下载的模型
echo "=================================="
echo "下载完成！已下载的模型:"
echo "=================================="
ls -lh models/
echo ""

echo "=================================="
echo "后续步骤:"
echo "=================================="
echo "1. 安装依赖包:"
echo "   pip install -r vision_requirements.txt"
echo ""
echo "2. 安装 Grounding DINO:"
echo "   cd GroundingDINO"
echo "   pip install -e ."
echo "   cd .."
echo ""
echo "3. 安装 SAM:"
echo "   pip install git+https://github.com/facebookresearch/segment-anything.git"
echo ""
echo "4. 运行示例:"
echo "   python vision_example.py"
echo ""
echo "完成！"

