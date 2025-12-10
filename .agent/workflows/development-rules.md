---
description: Development rules and conventions for stone-detection-service project
---

# Stone Detection Service - Development Rules

## Project Structure Rules

When working on this project, follow this structure:

1. **Source Code** → `src/`
   - Models go in `src/models/`
   - Inference engines go in `src/inference/`
   - Training scripts go in `src/training/`
   - Utility functions go in `src/utils/`

2. **Scripts** → `scripts/`
   - Runnable entry-point scripts only
   - Must import from `src/` packages

3. **Tools** → `tools/`
   - Data preparation and analysis tools
   - Dataset download/conversion scripts

4. **Configuration** → `config/`
   - YAML config files
   - Environment templates

5. **Data** → `data/`
   - `data/raw/` - Raw images
   - `data/processed/` - Processed data
   - `data/annotations/` - Label files (COCO JSON format)
   - `data/datasets/` - Complete dataset folders

6. **Results** → `results/`
   - Inference outputs
   - Visualizations
   - Analysis reports

## Coding Conventions

### Python Style
- Use type hints for function parameters and returns
- Use docstrings for all classes and public functions
- Follow PEP 8 naming conventions
- Maximum line length: 100 characters

### Imports
```python
# Standard library
import os
import json

# Third-party
import torch
import numpy as np

# Local
from src.models import SEUNet
from src.utils import visualization
```

### Model Development
- All models inherit from `torch.nn.Module`
- Include `forward()` method with type hints
- Document input/output tensor shapes in docstring

### Inference Scripts
- Support both CPU and GPU execution
- Include progress logging
- Save results to `results/` directory
- Use config files from `config/` for parameters

## File Naming

| Type | Pattern | Example |
|------|---------|---------|
| Models | `snake_case.py` | `se_unet.py` |
| Scripts | `run_*.py` or `main.py` | `run_inference.py` |
| Tools | `*_tool.py` or descriptive | `dataset_download.py` |
| Config | `config*.yaml` | `config.yaml` |
| Requirements | `requirements*.txt` | `requirements.mobilesam.txt` |

## Git Workflow

1. Never commit:
   - Model weights (`.pt`, `.pth`)
   - Large datasets
   - Log files
   - Result files

2. Always commit:
   - Source code changes
   - Configuration updates
   - Documentation changes
   - Requirement updates

## Testing

- Unit tests go in `tests/`
- Name test files as `test_*.py`
- Run tests with: `pytest tests/`

## Common Commands

```bash
# Run main inference
python scripts/main.py

# Run async YOLO+MobileSAM
python scripts/run_async_yolo_mobilesam.py

# Check model availability
python scripts/check_models.py

# Monitor performance
python scripts/performance_monitor.py
```
