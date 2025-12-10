# Requirements Directory

This directory contains organized dependency files for the Stone Detection Service.

## Quick Start

```bash
# Install core dependencies only
pip install -r requirements/requirements.txt

# Full installation (detection + segmentation)
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-detection.txt
pip install -r requirements/requirements-segmentation.txt
```

## Requirements Files

| File | Purpose |
|------|---------|
| `requirements.txt` | **Core dependencies** - PyTorch, NumPy, OpenCV, etc. |
| `requirements-detection.txt` | **Object Detection** - YOLO, Detectron2, Mask R-CNN |
| `requirements-segmentation.txt` | **Segmentation** - SAM, SAM2, MobileSAM, NanoSAM, FastSAM |
| `requirements-training.txt` | **Training** - U-Net, augmentation, metrics, experiment tracking |
| `requirements-api.txt` | **API Server** - FastAPI, async utilities, monitoring |
| `requirements-dev.txt` | **Development** - Testing, linting, documentation |

## Installation Order

1. **Install PyTorch with CUDA first** (not included in requirements.txt):
   ```bash
   # For CUDA 12.1
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install core requirements**:
   ```bash
   pip install -r requirements/requirements.txt
   ```

3. **Install task-specific requirements** based on your use case:
   
   - For **inference only**:
     ```bash
     pip install -r requirements/requirements-detection.txt
     pip install -r requirements/requirements-segmentation.txt
     ```
   
   - For **training**:
     ```bash
     pip install -r requirements/requirements-training.txt
     ```
   
   - For **API deployment**:
     ```bash
     pip install -r requirements/requirements-api.txt
     ```

## Model-Specific Dependencies

### SAM Variants (from GitHub)

These models are installed directly from GitHub:

```bash
# Original SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# MobileSAM (lightweight)
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# SAM2 (video support)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# NanoSAM (TensorRT)
pip install git+https://github.com/NVIDIA-AI-IOT/nanosam.git

# FastSAM (CNN-based)
pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git
```

### Detectron2 (from Facebook wheels)

```bash
# For CUDA 12.1, PyTorch 2.1
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html

# Or from source
pip install git+https://github.com/facebookresearch/detectron2.git
```

## Environment Recommendations

- **Python**: 3.9 - 3.11
- **CUDA**: 11.8 or 12.x
- **GPU**: 8GB+ VRAM recommended (16GB+ for training)
- **RAM**: 16GB minimum, 32GB recommended

## Troubleshooting

### CUDA Version Mismatch
Ensure PyTorch CUDA version matches your system CUDA:
```bash
python -c "import torch; print(torch.version.cuda)"
nvidia-smi
```

### ImportError for GitHub Packages
If a GitHub package fails to install, try:
```bash
pip install --upgrade pip
pip install git+https://github.com/USERNAME/REPO.git --no-cache-dir
```
