---
description: How to run inference on images
---

# Running Inference

## Quick Start

// turbo
```bash
python scripts/main.py
```

## Using YOLO + MobileSAM Ensemble

1. **Ensure models are available**:
   // turbo
   ```bash
   python scripts/check_models.py
   ```

2. **Run async processing**:
   ```bash
   python scripts/run_async_yolo_mobilesam.py
   ```

3. **Check results** in `results/` folder

## Configuration

Edit `config/config.yaml` to adjust:
- Model paths
- Processing parameters
- Output settings

## Input Images

Place images in:
- `data/raw/` for new images
- Or specify path in config

## Output

Results are saved to `results/`:
- `results/visualizations/` - Visual outputs
- `results/json_output/` - Detection data
- `results/reports/` - Analysis reports
