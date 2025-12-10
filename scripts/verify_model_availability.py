#!/usr/bin/env python3
"""
Verify Model Availability

Check which YOLO and other models are available for stone detection.
Uses centralized configuration from config/config.yaml.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from src.utils.settings import config


def check_model_file(name: str, path: str) -> bool:
    """Check if a model file exists."""
    full_path = project_root / path
    exists = full_path.exists()
    
    if exists:
        size_mb = full_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {name}: {path} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {name}: {path} (NOT FOUND)")
    
    return exists


def check_gpu_availability() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  ⚠️  GPU: Not available (will use CPU)")
        
        return has_gpu
        
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False


def check_dependencies() -> dict:
    """Check required dependencies."""
    dependencies = {
        'torch': False,
        'torchvision': False,
        'ultralytics': False,
        'cv2': False,
        'numpy': False,
        'PIL': False,
        'yaml': False,
    }
    
    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'PIL':
                from PIL import Image
            elif dep == 'yaml':
                import yaml
            else:
                __import__(dep)
            dependencies[dep] = True
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep}")
    
    return dependencies


def main():
    """Main function to check model and system availability."""
    print("=" * 60)
    print("STONE DETECTION - MODEL & SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Check GPU
    print("\n📊 GPU STATUS:")
    has_gpu = check_gpu_availability()
    
    # Check dependencies
    print("\n📦 DEPENDENCIES:")
    deps = check_dependencies()
    
    # Check model files from config
    print("\n🤖 MODEL FILES:")
    
    models_ok = True
    
    # YOLO model
    yolo_path = config.get("models.yolo.path")
    if yolo_path:
        if not check_model_file("YOLO", yolo_path):
            models_ok = False
    
    # MobileSAM model
    sam_path = config.get("models.mobilesam.path")
    if sam_path:
        if not check_model_file("MobileSAM", sam_path):
            models_ok = False
    
    # U-Net model
    unet_path = config.get("models.unet.path")
    if unet_path:
        check_model_file("U-Net", unet_path)
    
    # SE-UNet model
    se_unet_path = config.get("models.se_unet.path")
    if se_unet_path:
        check_model_file("SE-UNet", se_unet_path)
    
    # Check directories from config
    print("\n📁 DIRECTORIES:")
    
    dirs_to_check = [
        ("Data (raw)", config.get("paths.raw_images")),
        ("Data (annotations)", config.get("paths.annotations")),
        ("Results", config.get("paths.results_dir")),
        ("Logs", config.get("paths.logs_dir")),
        ("Weights", config.get("paths.weights_dir")),
    ]
    
    for name, path in dirs_to_check:
        if path:
            dir_path = Path(path)
            if dir_path.exists():
                print(f"  ✅ {name}: {path}")
            else:
                print(f"  ⚠️  {name}: {path} (will be created)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_deps_ok = all(deps.values())
    
    if all_deps_ok and models_ok and has_gpu:
        print("🚀 System is fully ready for GPU-accelerated inference!")
    elif all_deps_ok and models_ok:
        print("✅ System is ready for CPU inference")
        print("   (Install CUDA for faster processing)")
    elif all_deps_ok:
        print("⚠️  Dependencies OK but some models are missing")
        print("   Download models to the weights/ directory")
    else:
        print("❌ Some dependencies are missing")
        print("   Run: pip install -r requirements/requirements.txt")
    
    print("\n💡 NEXT STEPS:")
    if not models_ok:
        print("   1. Download YOLO model (best.pt) to weights/")
        print("   2. Download MobileSAM model (mobile_sam.pt) to weights/")
    print("   3. Run inference: python scripts/ensemble_yolo_mobilesam_inference.py")
    print("   4. Or traditional: python scripts/traditional_segmentation.py --input <image>")


if __name__ == '__main__':
    main()
