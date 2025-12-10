import json
import os
import cv2
import numpy as np
from PIL import Image
import argparse
from pycocotools.coco import COCO
from tqdm.auto import tqdm

def ensure_binary_mask(mask):
    """Преобразует маску в бинарный формат uint8"""
    return (mask > 0).astype(np.uint8)

def create_sam_mask(masks, height, width):
    """Создает финальную маску без вычитания границ между объектами"""
    # Инициализируем финальную маску
    final_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Обрабатываем каждую маску отдельно
    for mask in masks:
        # Просто объединяем маски
        final_mask = np.logical_or(final_mask, ensure_binary_mask(mask)).astype(np.uint8)
    
    # Конвертируем в формат SAM (0 - черный, 255 - белый)
    final_mask = final_mask * 255
    
    return final_mask

def main(args):
    # Create output directories if they don't exist
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "masks"), exist_ok=True)

    # Initialize COCO api
    coco = COCO(args.coco_json)

    # Get all image ids
    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} images in the dataset")

    for img_id in tqdm(img_ids):
        # Get image info
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(args.img_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue

        # Load and save the image
        img = Image.open(image_path)
        img.save(os.path.join(args.out_dir, "images", img_info['file_name']))

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            print(f"Warning: No annotations found for image {img_info['file_name']}")
            continue

        # Создаем список для хранения масок отдельных объектов
        instance_masks = []
        
        for ann in anns:
            # Получаем маску для каждой аннотации
            ann_mask = coco.annToMask(ann)
            instance_masks.append(ann_mask)

        # Создаем SAM маску из всех отдельных масок
        sam_mask = create_sam_mask(instance_masks, img_info['height'], img_info['width'])
        
        # Save the mask as PNG
        mask_filename = os.path.splitext(img_info['file_name'])[0] + ".png"
        cv2.imwrite(os.path.join(args.out_dir, "masks", mask_filename), sam_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO format dataset to SAM format")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to COCO JSON annotations file")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for the SAM dataset")
    
    args = parser.parse_args()
    main(args)
