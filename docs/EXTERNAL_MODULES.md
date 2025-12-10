# External Modules

This project uses the following external modules/repositories:

## Submodules

| Module | Path | Repository | Purpose |
|--------|------|------------|---------|
| **GroundingDINO** | `GroundingDINO/` | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Text-guided object detection |
| **RAM (Recognize Anything)** | `recognize-anything/` | [xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything) | Auto image tagging (RAM, RAM+, TAG2TEXT) |
| **SAM-HQ** | `sam-hq/` | [SysCV/sam-hq](https://github.com/SysCV/sam-hq) | High-quality segmentation |
| **Grounded-SAM** | `Grounded-Segment-Anything/` | [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Combined detection + segmentation |

## Model Weights

| Model | Weight File | Size | Download |
|-------|-------------|------|----------|
| GroundingDINO SwinT | `weight/groundingdino_swint_ogc.pth` | ~694 MB | [HuggingFace](https://huggingface.co/ShilongLiu/GroundingDINO) |
| SAM ViT-H | `weight/sam_vit_h_4b8939.pth` | ~2.6 GB | [Meta AI](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| RAM Swin-L | `weight/ram_swin_large_14m.pth` | ~1.7 GB | [HuggingFace](https://huggingface.co/xinyu1205/recognize_anything_model) |
| SAM-HQ ViT-H | `weight/sam_hq_vit_h.pth` | ~2.6 GB | [SysCV](https://github.com/SysCV/sam-hq#model-checkpoints) |

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/your-repo/stone-detection-service.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

## PYTHONPATH Setup

To use the modules, add them to PYTHONPATH:

```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;GroundingDINO;recognize-anything;sam-hq

# Linux/Mac
export PYTHONPATH=$PYTHONPATH:GroundingDINO:recognize-anything:sam-hq
```

---

## Additional Modules to Consider

These modules could improve our auto-labeling pipeline:

| Module | Repository | Purpose | Benefit |
|--------|------------|---------|---------|
| **SAM2** | [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2) | Next-gen SAM | Better masks, video support |
| **MobileSAM** | [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Lightweight SAM | 10x faster inference |
| **YOLO-World** | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) | Open-vocab YOLO | Real-time detection |
| **OWL-ViT / OWLv2** | [google-research/scenic](https://github.com/google-research/scenic) | Open-vocab detection | Good for small objects |
| **Florence-2** | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | Vision foundation | Multi-task (detect, segment, caption) |
| **DINOv2** | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Self-supervised features | Feature extraction |

### Why Add These?

1. **SAM2** - Better mask quality than SAM, especially for video
2. **MobileSAM** - Can run multiple passes faster for ensemble
3. **YOLO-World** - Alternative to GroundingDINO for detection
4. **Florence-2** - One model that can do detection + segmentation + captioning
5. **OWLv2** - Good at finding small objects that others miss
