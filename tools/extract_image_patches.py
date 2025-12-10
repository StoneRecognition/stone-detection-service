#!/usr/bin/env python3
"""
Extract Image Patches

Utility functions for extracting patches from volumetric data.
Used by download_dataset_patches.py and other tools.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict


def extract_patch(
    volume: np.ndarray,
    start_d: int,
    start_h: int,
    start_w: int,
    patch_size: Tuple[int, int, int]
) -> np.ndarray:
    """
    Extract a 3D patch from a volumetric image.
    
    Args:
        volume: 3D numpy array (D, H, W)
        start_d: Starting depth index
        start_h: Starting height index
        start_w: Starting width index
        patch_size: Tuple of (depth, height, width)
        
    Returns:
        Extracted patch
    """
    patch_d, patch_h, patch_w = patch_size
    return volume[start_d:start_d+patch_d, start_h:start_h+patch_h, start_w:start_w+patch_w]


def save_patch(
    patch_data: np.ndarray,
    patch_seg: np.ndarray,
    images_folder: str,
    masks_folder: str,
    patch_num: int
) -> None:
    """
    Save image and segmented mask patches to their respective folders.
    
    Args:
        patch_data: Image patch array
        patch_seg: Segmentation mask patch array
        images_folder: Path to save images
        masks_folder: Path to save masks
        patch_num: Patch number for file naming
    """
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    
    patch_data_filename = os.path.join(images_folder, f'{patch_num:05d}.npy')
    patch_seg_filename = os.path.join(masks_folder, f'{patch_num:05d}.npy')
    
    np.save(patch_data_filename, patch_data)
    np.save(patch_seg_filename, patch_seg)


def create_directories(data_dirs: Dict[str, str]) -> None:
    """
    Create directory structure for images and segmented masks.
    
    Args:
        data_dirs: Dictionary with 'images', 'masks', and 'folders' keys
    """
    images_dir = data_dirs["images"]
    masks_dir = data_dirs["masks"]
    folders = data_dirs.get("folders", ["Training Images", "Validation Images", "Test Images"])
    
    for folder in folders:
        os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(masks_dir, folder), exist_ok=True)


def display_patches(
    patch_data: np.ndarray,
    patch_seg: np.ndarray,
    patch_num: int,
    save_path: str = None
) -> None:
    """
    Display image and segmented mask patches side by side.
    
    Args:
        patch_data: Image patch array
        patch_seg: Segmentation mask patch array
        patch_num: Patch number for labeling
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Image
    im1 = axes[0].imshow(patch_data, cmap='gray')
    axes[0].set_title(f"Patch {patch_num} - Image")
    plt.colorbar(im1, ax=axes[0])
    
    # Mask
    im2 = axes[1].imshow(patch_seg, cmap='gray')
    axes[1].set_title(f"Patch {patch_num} - Mask")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def extract_2d_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = None,
    min_mask_pixels: int = 10
) -> list:
    """
    Extract 2D patches from an image and mask pair.
    
    Args:
        image: 2D or 3D image array (H, W) or (H, W, C)
        mask: 2D mask array (H, W)
        patch_size: Size of square patches
        stride: Stride between patches (default: patch_size)
        min_mask_pixels: Minimum mask pixels to include patch
        
    Returns:
        List of (image_patch, mask_patch) tuples
    """
    if stride is None:
        stride = patch_size
    
    h, w = image.shape[:2]
    patches = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Filter empty patches
            if np.sum(mask_patch > 127) >= min_mask_pixels:
                patches.append((img_patch, mask_patch))
    
    return patches


if __name__ == "__main__":
    # Test patch extraction
    print("Testing patch extraction utilities...")
    
    # Create test volume
    test_volume = np.random.randint(0, 256, (100, 100, 100), dtype=np.uint8)
    
    # Extract patch
    patch = extract_patch(test_volume, 10, 20, 30, (32, 32, 32))
    print(f"Extracted patch shape: {patch.shape}")
    
    # Test 2D extraction
    test_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    test_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    
    patches = extract_2d_patches(test_image, test_mask, patch_size=128, stride=64)
    print(f"Extracted {len(patches)} 2D patches")
    
    print("✅ Test passed!")
