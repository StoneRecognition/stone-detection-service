#!/usr/bin/env python3
"""
Download Dataset Patches

Downloads volumetric rock images from Digital Rocks Portal and extracts patches.
Uses centralized utilities from src/.
"""

import sys
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from centralized modules
from config.digital_rocks_sources import IMAGE_LINKS, AVAILABLE_IMAGES, DIMENSIONS, PATCH_SIZE, DATA_DIRS
from src.utils.dataloader import load_raw_file, download_file


def extract_patch(volume, start_d, start_h, start_w, patch_size):
    """Extract a 3D patch from a volume."""
    patch_d, patch_h, patch_w = patch_size
    return volume[start_d:start_d+patch_d, start_h:start_h+patch_h, start_w:start_w+patch_w]


def save_patch(patch_data, patch_seg, images_folder, masks_folder, patch_num):
    """Save image and mask patches as NPY files."""
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    
    np.save(os.path.join(images_folder, f'{patch_num:05d}.npy'), patch_data)
    np.save(os.path.join(masks_folder, f'{patch_num:05d}.npy'), patch_seg)


def create_directories(data_dirs):
    """Create directory structure for dataset."""
    images_dir = data_dirs["images"]
    masks_dir = data_dirs["masks"]
    
    for folder in data_dirs["folders"]:
        os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(masks_dir, folder), exist_ok=True)


def display_patches(patch_data, patch_seg, patch_num):
    """Display image and mask patches side by side."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(patch_data[:, :], cmap='gray')
    plt.title(f"Patch {patch_num} - Image")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(patch_seg[:, :], cmap='gray')
    plt.title(f"Patch {patch_num} - Mask")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to download and extract patches."""
    print("=" * 60)
    print("Digital Rocks Portal - Dataset Patch Downloader")
    print("=" * 60)
    
    print("\nAvailable images:", ", ".join(AVAILABLE_IMAGES))
    selected_image = input("Enter image name (e.g., Berea): ").strip()
    
    if selected_image not in IMAGE_LINKS:
        print(f"Error: Invalid image '{selected_image}'")
        print(f"Available: {AVAILABLE_IMAGES}")
        return
    
    num_train = int(input("Number of training patches: "))
    num_val = int(input("Number of validation patches: "))
    num_test = int(input("Number of test patches: "))
    
    # Get image info
    image_info = IMAGE_LINKS[selected_image]
    n_image = int(image_info["number"])
    random_seed = image_info["seed"]
    
    # File names
    original_file = f'{selected_image}_2d25um_grayscale_filtered.raw'
    segmented_file = f'{selected_image}_2d25um_binary.raw'
    
    # Download files if needed
    if not os.path.exists(original_file):
        print(f"\nDownloading {selected_image} original data...")
        download_file(image_info["original"], original_file)
    else:
        print(f"Found local file: {original_file}")
    
    if not os.path.exists(segmented_file):
        print(f"Downloading {selected_image} segmented data...")
        download_file(image_info["segmented"], segmented_file)
    else:
        print(f"Found local file: {segmented_file}")
    
    # Load data
    shape = (DIMENSIONS["depth"], DIMENSIONS["height"], DIMENSIONS["width"])
    print(f"\nLoading data with shape {shape}...")
    
    data = load_raw_file(original_file, shape)
    data_seg = load_raw_file(segmented_file, shape)
    
    print(f"Original data shape: {data.shape}")
    print(f"Segmented data shape: {data_seg.shape}")
    
    # Create directories
    create_directories(DATA_DIRS)
    
    # Extract patches
    total_patches = num_train + num_val + num_test
    print(f"\nExtracting {total_patches} patches...")
    
    random.seed(random_seed)
    
    for i in tqdm(range(1, total_patches + 1)):
        d, h, w = data.shape
        patch_d, patch_h, patch_w = PATCH_SIZE
        
        start_d = random.randint(0, d - patch_d)
        start_h = random.randint(0, h - patch_h)
        start_w = random.randint(0, w - patch_w)
        
        patch_data = extract_patch(data, start_d, start_h, start_w, PATCH_SIZE).squeeze()
        patch_seg = extract_patch(data_seg, start_d, start_h, start_w, PATCH_SIZE).squeeze()
        
        # Determine folder and patch number
        if i <= num_train:
            folder = 'Training Images'
            patch_num = i + int(num_train * (n_image - 1))
        elif i <= num_train + num_val:
            folder = 'Validation Images'
            patch_num = i - num_train + int(num_val * (n_image - 1))
        else:
            folder = 'Test Images'
            patch_num = i - num_train - num_val + int(num_test * (n_image - 1))
        
        images_folder = os.path.join(DATA_DIRS["images"], folder)
        masks_folder = os.path.join(DATA_DIRS["masks"], folder)
        
        save_patch(patch_data, patch_seg, images_folder, masks_folder, patch_num)
        
        # Display first few patches
        if i <= 3:
            display_patches(patch_data, patch_seg, i)
    
    print(f"\n✅ Extracted {num_train} training, {num_val} validation, {num_test} test patches")
    print(f"Saved to: {DATA_DIRS['images']}")


if __name__ == "__main__":
    main()
