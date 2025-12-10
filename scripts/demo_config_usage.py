"""
Example: How to Use the Configuration System

This script demonstrates how to use the centralized configuration
instead of hardcoding values in your scripts.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.settings import (
    config, 
    get_model_path, 
    get_output_dir, 
    get_data_dir,
    ensure_dirs
)


def main():
    """Demonstrate configuration usage."""
    
    print("=" * 60)
    print("Stone Detection Service - Configuration Example")
    print("=" * 60)
    
    # =========================================================================
    # 1. Accessing configuration with dot notation
    # =========================================================================
    print("\n1. DOT NOTATION ACCESS:")
    print(f"   Project root: {config.project_root}")
    print(f"   YOLO model path: {config.models.yolo.path}")
    print(f"   YOLO confidence: {config.models.yolo.confidence_threshold}")
    print(f"   Training epochs: {config.training.epochs}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   GPU memory fraction: {config.hardware.gpu.memory_fraction}")
    
    # =========================================================================
    # 2. Using get() method with defaults
    # =========================================================================
    print("\n2. GET WITH DEFAULTS:")
    lr = config.get("training.learning_rate", default=0.001)
    print(f"   Learning rate: {lr}")
    
    # Safe access to potentially missing config
    custom_value = config.get("custom.missing_key", default="fallback")
    print(f"   Custom value (with fallback): {custom_value}")
    
    # =========================================================================
    # 3. Getting full sections
    # =========================================================================
    print("\n3. GET SECTIONS:")
    yolo_config = config.get_section("models.yolo")
    print(f"   YOLO config: {yolo_config}")
    
    gpu_config = config.get_section("hardware.gpu")
    print(f"   GPU config: {gpu_config}")
    
    # =========================================================================
    # 4. Using helper functions
    # =========================================================================
    print("\n4. HELPER FUNCTIONS:")
    
    # Get model paths (returns absolute Path objects)
    yolo_path = get_model_path("yolo")
    mobilesam_path = get_model_path("mobilesam")
    print(f"   YOLO path: {yolo_path}")
    print(f"   MobileSAM path: {mobilesam_path}")
    
    # Get output directories
    results_dir = get_output_dir()
    viz_dir = get_output_dir("visualizations")
    print(f"   Results dir: {results_dir}")
    print(f"   Visualizations dir: {viz_dir}")
    
    # Get data directories
    raw_data = get_data_dir("raw_images")
    annotations = get_data_dir("annotations")
    print(f"   Raw images: {raw_data}")
    print(f"   Annotations: {annotations}")
    
    # =========================================================================
    # 5. Ensure directories exist
    # =========================================================================
    print("\n5. ENSURE DIRECTORIES:")
    ensure_dirs()
    print("   All output directories created/verified!")
    
    # =========================================================================
    # 6. Example: Using in a training script
    # =========================================================================
    print("\n6. TRAINING SCRIPT EXAMPLE:")
    print(f"""
    # Instead of hardcoding:
    # epochs = 100
    # batch_size = 16
    # lr = 0.001
    
    # Use configuration:
    epochs = config.training.epochs           # {config.training.epochs}
    batch_size = config.training.batch_size   # {config.training.batch_size}
    lr = config.training.learning_rate        # {config.training.learning_rate}
    """)
    
    # =========================================================================
    # 7. Example: Using in an inference script
    # =========================================================================
    print("7. INFERENCE SCRIPT EXAMPLE:")
    print(f"""
    # Load model with config path
    model_path = get_model_path("yolo")
    
    # Use inference settings
    device = config.inference.device             # {config.inference.device}
    batch_size = config.inference.batch_size     # {config.inference.batch_size}
    save_masks = config.inference.output.save_masks  # {config.inference.output.save_masks}
    
    # GPU optimization
    if config.hardware.gpu.mixed_precision:      # {config.hardware.gpu.mixed_precision}
        # Enable mixed precision training
        pass
    """)
    
    print("=" * 60)
    print("Configuration system ready to use!")
    print("=" * 60)


if __name__ == "__main__":
    main()
