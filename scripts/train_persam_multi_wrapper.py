#!/usr/bin/env python3
"""
Wrapper to Run PerSAM-F Multi-Object Training.
1. Converts COCO dataset if needed (Direct Import).
2. Runs Personalize-SAM/persam_f_multi_obj.py with correct paths (Subprocess).
"""

import os
import sys
import subprocess
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PERSAM_DIR = PROJECT_ROOT / "Personalize-SAM"
DATASET_FORMATTED_DIR = PROJECT_ROOT / "data/datasets/PerSam-F_26_Formatted"
SAM_CHECKPOINT = PROJECT_ROOT / "weight/mobile_sam.pt"

# Add scripts dir to path to import converter
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    import convert_coco_to_persam
except ImportError:
    print("Error: Could not import convert_coco_to_persam.py")
    sys.exit(1)

def run_conversion():
    print("--- Step 1: Converting Dataset (Internal) ---")
    try:
        convert_coco_to_persam.convert()
        return True
    except Exception as e:
        print(f"Conversion failed with error: {e}")
        return False

def run_training():
    print("\n--- Step 2: Running PerSAM-F Training ---")
    
    train_script = "persam_f_multi_obj.py"
    
    # Using python executable from current env
    python_exe = sys.executable
    
    cmd = [
        python_exe,
        train_script,
        "--data", str(DATASET_FORMATTED_DIR),
        "--outdir", "persam_f_stone_custom",
        "--ckpt", str(SAM_CHECKPOINT),
        "--sam_type", "vit_t",
        "--train_epoch_outside", "2", 
        "--train_epoch_inside", "100"
    ]
    
    print(f"Running in {PERSAM_DIR}: {' '.join(cmd)}")
    
    # We must run inside Personalize-SAM dir because of its relative imports
    subprocess.call(cmd, cwd=str(PERSAM_DIR))

def main():
    if not DATASET_FORMATTED_DIR.exists():
        if not run_conversion():
            return
    else:
        print(f"Dataset already exists at {DATASET_FORMATTED_DIR}")
            
    run_training()

if __name__ == "__main__":
    main()
