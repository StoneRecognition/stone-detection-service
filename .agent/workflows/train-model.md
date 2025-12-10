---
description: How to train models in this project
---

# Training Models

## Available Training Scripts

1. **U-Net Training**:
   ```bash
   python src/training/train.py
   ```

2. **SE-PP-UNet Training**:
   ```bash
   python src/training/train_se_pp_unet.py
   ```

3. **Custom Dataset Training**:
   ```bash
   python src/training/train_custom_dataset.py
   ```

## Prepare Data

1. Place training images in `data/raw/`
2. Place annotations in `data/annotations/` (COCO format)
3. Update config in `config/config.yaml`

## Requirements

Install training dependencies:
// turbo
```bash
pip install -r requirements/requirements.unet.txt
```

## GPU Optimization

For 16GB VRAM GPUs, the system uses:
- Mixed precision (FP16)
- 95% GPU memory allocation
- Batch processing

## Output

- Trained weights saved to `weights/`
- Training logs to `logs/`
