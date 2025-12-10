"""
COCO Annotation Utilities Module

Utilities for handling COCO format annotations.
"""

import cv2
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

from .json_utils import save_json, load_json


def load_coco_annotations(json_path: Union[str, Path]) -> Dict:
    """
    Load COCO format annotations from file.
    
    Args:
        json_path: Path to COCO JSON file
        
    Returns:
        COCO annotation dictionary
    """
    return load_json(json_path)


def load_or_create_coco_dataset(
    coco_path: Union[str, Path],
    default_categories: Optional[List[Dict]] = None
) -> Tuple[List[Dict], List[Dict], List[Dict], int, Dict[str, int]]:
    """
    Load existing COCO dataset or create empty structure.
    
    Args:
        coco_path: Path to existing COCO JSON file
        default_categories: Default categories if creating new dataset
        
    Returns:
        Tuple of (images, annotations, categories, next_ann_id, file2id_map)
    """
    coco_path = Path(coco_path)
    
    if default_categories is None:
        default_categories = [{"id": 1, "name": "object", "supercategory": "object"}]
    
    if coco_path.exists():
        coco_data = load_json(coco_path)
        images = coco_data.get('images', [])
        annotations = coco_data.get('annotations', [])
        categories = coco_data.get('categories', default_categories)
        next_ann_id = max([ann['id'] for ann in annotations], default=0) + 1
        file2id = {img['file_name'].split('.')[0]: img['id'] for img in images}
    else:
        images = []
        annotations = []
        categories = default_categories
        next_ann_id = 1
        file2id = {}
    
    return images, annotations, categories, next_ann_id, file2id


def save_dataset_metadata(
    results: List[Dict],
    output_path: Union[str, Path],
    extra_info: Optional[Dict] = None
) -> None:
    """
    Save dataset processing metadata.
    
    Args:
        results: List of result metadata dictionaries
        output_path: Path to save metadata JSON
        extra_info: Optional additional metadata to include
    """
    dataset_metadata = {
        "total_images": len(results),
        "dataset_info": results
    }
    if extra_info:
        dataset_metadata.update(extra_info)
    save_json(dataset_metadata, output_path)


def save_coco_annotations(
    images: List[Dict],
    annotations: List[Dict],
    categories: List[Dict],
    output_path: Union[str, Path]
) -> None:
    """
    Save annotations in COCO format.
    
    Args:
        images: List of image entries
        annotations: List of annotation entries
        categories: List of category entries
        output_path: Output JSON path
    """
    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    save_json(dataset, output_path)


def create_coco_annotation_from_mask(
    mask: np.ndarray,
    ann_id: int,
    image_id: int,
    category_id: int = 1,
    min_area: int = 100,
    min_contour_points: int = 6
) -> Optional[Dict]:
    """
    Create a single COCO annotation entry from a binary mask.
    
    Args:
        mask: Binary mask (2D numpy array)
        ann_id: Annotation ID
        image_id: Image ID this annotation belongs to
        category_id: Category ID (default: 1)
        min_area: Minimum area threshold for valid annotation
        min_contour_points: Minimum contour points for valid segmentation
        
    Returns:
        COCO annotation dictionary or None if mask doesn't meet criteria
    """
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < min_contour_points:
        return None
    
    segmentation = largest_contour.flatten().tolist()
    area = cv2.contourArea(largest_contour)
    
    if area < min_area:
        return None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [segmentation],
        "area": float(area),
        "bbox": [float(x), float(y), float(w), float(h)],
        "iscrowd": 0
    }


def create_coco_annotations_from_masks(
    masks: List[np.ndarray],
    start_ann_id: int,
    image_id: int,
    category_id: int = 1,
    min_area: int = 100,
    min_contour_points: int = 6,
    bboxes: Optional[List[List[float]]] = None
) -> Tuple[List[Dict], int]:
    """
    Create COCO annotations from a list of masks.
    
    Args:
        masks: List of binary masks
        start_ann_id: Starting annotation ID
        image_id: Image ID these annotations belong to
        category_id: Category ID (default: 1)
        min_area: Minimum area threshold
        min_contour_points: Minimum contour points
        bboxes: Optional pre-computed bounding boxes (uses mask bbox if None)
        
    Returns:
        Tuple of (list of annotation dicts, next available ann_id)
    """
    annotations = []
    ann_id = start_ann_id
    
    for i, mask in enumerate(masks):
        mask_bin = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < min_contour_points:
                continue
            
            segmentation = contour.flatten().tolist()
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            # Use pre-computed bbox if provided, otherwise compute from contour
            if bboxes is not None and i < len(bboxes):
                x, y, w, h = bboxes[i]
            else:
                x, y, w, h = cv2.boundingRect(contour)
            
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],
                "area": float(area),
                "bbox": [float(x), float(y), float(w), float(h)],
                "iscrowd": 0
            })
            ann_id += 1
    
    return annotations, ann_id


def create_coco_image_entry(
    image_id: int,
    width: int,
    height: int,
    file_name: str
) -> Dict:
    """
    Create a COCO image entry.
    
    Args:
        image_id: Unique image ID
        width: Image width in pixels
        height: Image height in pixels
        file_name: Image filename
        
    Returns:
        COCO image entry dictionary
    """
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name
    }


def build_mask_from_coco(
    coco_json: Dict,
    image_id: int,
    shape: Tuple[int, int]
) -> np.ndarray:
    """
    Build a binary mask from COCO annotations for a given image.
    
    Args:
        coco_json: COCO annotation dictionary
        image_id: ID of the image
        shape: Output mask shape (height, width)
        
    Returns:
        Binary mask array
    """
    mask = np.zeros(shape, dtype=np.uint8)
    anns = [ann for ann in coco_json.get('annotations', []) 
            if ann['image_id'] == image_id]
    
    for ann in anns:
        for seg in ann.get('segmentation', []):
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
    
    return mask


def draw_coco_overlay(
    coco_json: Dict,
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    alpha: float = 0.3,
    contour_thickness: int = 2
) -> None:
    """
    Draw COCO annotations as overlay on images.
    
    Args:
        coco_json: COCO annotation dictionary
        images_dir: Directory containing images
        output_dir: Directory to save overlay images
        alpha: Transparency for mask overlay
        contour_thickness: Thickness of contour lines
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path(images_dir)
    id2file = {img['id']: img['file_name'] for img in coco_json['images']}
    
    for img_id, file_name in id2file.items():
        img_path = images_dir / file_name
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found")
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
        
        mask = np.zeros(img.shape[:2], np.uint8)
        color_mask = np.zeros_like(img)
        anns = [ann for ann in coco_json['annotations'] if ann['image_id'] == img_id]
        
        rng = np.random.default_rng(42)
        for ann in anns:
            segs = ann.get('segmentation', [])
            color = tuple(int(x) for x in rng.integers(0, 255, size=3))
            for seg in segs:
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 255)
                cv2.fillPoly(color_mask, [pts], color)
        
        # Create overlay
        overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            color = tuple(int(x) for x in rng.integers(0, 255, size=3))
            cv2.drawContours(overlay, [cnt], -1, color, contour_thickness)
        
        # Save
        out_path = output_dir / f"{Path(file_name).stem}_overlay.png"
        cv2.imwrite(str(out_path), overlay)
        
        # Memory cleanup
        del img, mask, color_mask, overlay
        gc.collect()
