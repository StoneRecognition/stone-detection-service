---
description: How to run the PerSAM-F automated stone detection pipeline
---

# PerSAM-F Stone Detection Pipeline

Automated 3-phase pipeline for generating stone detection datasets without manual labeling.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements/requirements-segmentation.txt
   pip install -r requirements/requirements-persam.txt
   ```

2. Download model weights:
   - SAM ViT-H: Already in `weight/sam_vit_h.pt`
   - Grounding DINO: Download `groundingdino_swint_ogc.pth` from https://github.com/IDEA-Research/GroundingDINO/releases and place in `weights/`

## Full Pipeline

Run all 3 phases automatically:

// turbo-all
```bash
python scripts/run_persam_pipeline.py --input data/raw --output results/persam_dataset
```

This will:
1. Generate reference mask from the first image
2. Fine-tune PerSAM-F in ~10 seconds
3. Process all images and generate COCO dataset

## Individual Phases

### Phase 1: Generate Reference Mask

```bash
python scripts/run_persam_pipeline.py --phase 1 --input data/raw/sample.jpg --output results/reference --prompt "stone contaminant"
```

Outputs:
- `sample_reference.png` - Reference image
- `sample_mask.png` - Segmentation mask
- `sample_overlay.png` - Visualization
- `sample_metadata.json` - Detection info

### Phase 2: Fine-tune PerSAM-F

```bash
python scripts/run_persam_pipeline.py --phase 2 --ref-image results/reference/sample_reference.png --ref-mask results/reference/sample_mask.png --iterations 1000
```

Training completes in ~10 seconds. Weights saved to `weights/persam_stone.pth`.

### Phase 3: Batch Detection

```bash
python scripts/run_persam_pipeline.py --phase 3 --input data/raw --output results/dataset --weights weights/persam_stone.pth
```

Outputs in `results/dataset/`:
- `annotations.json` - COCO format annotations
- `detections.json` - Detection coordinates (x, y, bbox)
- `masks/` - Individual segmentation masks
- `visualizations/` - Overlay images

## Configuration

Edit `config/config.yaml` under `persam:` section:

```yaml
persam:
  reference:
    text_prompt: "stone contaminant"
    confidence_threshold: 0.3
  training:
    iterations: 1000
    sam_type: "vit_h"  # or vit_t for MobileSAM
  inference:
    points_per_side: 32
    iou_threshold: 0.7
```

## Using MobileSAM (for Edge Devices)

For Jetson Nano or faster inference:

```bash
python scripts/run_persam_pipeline.py --input data/raw --output results/mobile --sam-type vit_t
```
