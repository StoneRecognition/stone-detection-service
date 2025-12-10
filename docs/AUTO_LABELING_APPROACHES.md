# Auto-Labeling Approaches Documentation

This document describes all the stone detection approaches developed and tested.

## Overview

We developed multiple approaches for automated stone detection and labeling:

| Approach | Script | Detections | Notes |
|----------|--------|------------|-------|
| GroundingDINO + SAM | `diagnose_step_by_step.py` | 13 masks | Box-based, misses small stones |
| SAM Auto Mask | `diagnose_sam_auto.py` | 100+ masks | Finds all objects automatically |

---

## Approach 1: GroundingDINO Box-Based Detection

**Script:** `scripts/diagnose_step_by_step.py`

### Pipeline
1. **RAM Auto-Tagging** - Discovers: `stone`, `rocky`, `rubble`
2. **GroundingDINO Detection** - Multi-prompt: stone, rock, pebble, rocky, rubble
3. **NMS** - 29 raw boxes → 14 merged boxes
4. **Size Filter** - 14 → 13 (remove full-image detections)
5. **SAM Segmentation** - Generate masks for each box

### Results
- Found **13 stones**
- Large objects detected well
- **Problem:** Misses ~90% of smaller stones

### Output Files
```
outputs/step_by_step/
├── step1_original.jpg           # Original image
├── step2_ram_tags.txt           # RAM detected tags
├── step3_stone_boxes.jpg        # Per-prompt detection (stone)
├── step3_rock_boxes.jpg         # Per-prompt detection (rock)
├── step3_pebble_boxes.jpg       # Per-prompt detection (pebble)
├── step3_rocky_boxes.jpg        # Per-prompt detection (rocky)
├── step3_rubble_boxes.jpg       # Per-prompt detection (rubble)
├── step3_all_raw.json           # All raw box data
├── step4_after_nms.jpg          # After NMS merge
├── step5_after_filter.jpg       # After size filter
├── step6_mask_1.jpg ... 13.jpg  # Individual SAM masks
└── step7_final_result.jpg       # Combined final result
```

---

## Approach 2: SAM Automatic Mask Generation

**Script:** `scripts/diagnose_sam_auto.py`

### Pipeline
1. **SAM Auto Mask Generator** - Grid-based point prompts
2. **Filter by Size** - Remove background (>30% image)
3. **Output all masks** - For review

### Configuration
```python
POINTS_PER_SIDE = 32       # Grid density
PRED_IOU_THRESH = 0.86     # Prediction quality
STABILITY_SCORE_THRESH = 0.92  # Mask stability
MIN_MASK_AREA = 500        # Minimum pixels
```

### Results
- Found **100+ potential objects** (masks)
- Top 20 masks saved with area info
- Filtered masks shown in `step3_filtered_masks.jpg`

### Output Files
```
outputs/sam_auto/
├── step1_original.jpg           # Original image
├── step2_all_masks.jpg          # All masks visualized
├── step2_mask_01_area*.jpg      # Individual masks (top 20)
├── step3_filtered_masks.jpg     # After size filter
└── mask_data.json               # Mask metadata
```

---

## Model Weights Used

| Model | Weight File | Purpose |
|-------|-------------|---------|
| GroundingDINO | `weight/groundingdino_swint_ogc.pth` | Text-guided detection |
| SAM ViT-H | `weight/sam_vit_h_4b8939.pth` | Segmentation |
| RAM | `weight/ram_swin_large_14m.pth` | Auto-tagging |

---

## Detection Parameters

### GroundingDINO
| Parameter | Value | Effect |
|-----------|-------|--------|
| BOX_THRESHOLD | 0.25 | Lower = more detections |
| TEXT_THRESHOLD | 0.25 | Lower = more matches |
| MAX_BOX_RATIO | 0.4 | Filter boxes > 40% of image |
| MIN_BOX_RATIO | 0.02 | Filter boxes < 2% of image |

### SAM Auto
| Parameter | Value | Effect |
|-----------|-------|--------|
| POINTS_PER_SIDE | 32 | More points = more masks |
| PRED_IOU_THRESH | 0.86 | Lower = more masks |
| MIN_MASK_AREA | 500 | Minimum mask size |

---

## Comparison

| Metric | GroundingDINO | SAM Auto |
|--------|---------------|----------|
| Total detections | 13 | 100+ |
| Large objects | ✅ Good | ✅ Good |
| Small objects | ❌ Misses | ✅ Finds |
| Classification | ✅ Stone labels | ❌ No labels |
| Speed | Fast | Slower |

### Recommendation
Combine both approaches:
1. Use **SAM Auto** to find all potential objects
2. Use **GroundingDINO** to classify which are stones
3. Apply **hierarchical filtering** by size levels
