#!/usr/bin/env python3
"""
Prepare PerSAM Dataset

Filters a large COCO JSON annotation file to only include images present in a target folder.
"""

import os
import json
import shutil
from pathlib import Path

# Configuration
DATASET_ROOT = "data/datasets/PerSam-F 26"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
INPUT_JSON = os.path.join(DATASET_ROOT, "instances_default.json")
OUTPUT_JSON = os.path.join(DATASET_ROOT, "instances_filtered.json")


def filter_dataset():
    print(f"Reading {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        coco = json.load(f)
        
    print(f"Original Dataset: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    
    # Get list of existing image files
    existing_files = set(os.listdir(IMAGES_DIR))
    print(f"Found {len(existing_files)} files in {IMAGES_DIR}")
    
    # Filter images
    valid_image_ids = set()
    filtered_images = []
    
    for img in coco['images']:
        file_name = os.path.basename(img['file_name'])
        if file_name in existing_files:
            valid_image_ids.add(img['id'])
            filtered_images.append(img)
            
    print(f"Filtered Images: {len(filtered_images)}")
    
    if len(filtered_images) == 0:
        print("WARNING: No matching images found! Check file names.")
        return
        
    # Filter annotations
    filtered_annotations = []
    for ann in coco['annotations']:
        if ann['image_id'] in valid_image_ids:
            filtered_annotations.append(ann)
            
    print(f"Filtered Annotations: {len(filtered_annotations)}")
    
    # Construct new COCO object
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco.get("categories", [])
    }
    
    # Save
    print(f"Saving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(new_coco, f, indent=2)
        
    print("Done!")
    
    # Verify mapping
    print("\nSample mapping check:")
    for img in filtered_images[:3]:
        anns = [a for a in filtered_annotations if a['image_id'] == img['id']]
        print(f"  {img['file_name']} (ID {img['id']}): {len(anns)} annotations")


if __name__ == "__main__":
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found")
        sys.exit(1)
        
    filter_dataset()
