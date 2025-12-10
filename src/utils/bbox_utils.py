"""
Bounding Box Utilities Module

Utilities for bounding box operations.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def get_bbox_from_mask(mask: np.ndarray, min_area: int = 100) -> Optional[List[float]]:
    """
    Get bounding box from a binary mask.
    
    Args:
        mask: Binary mask
        min_area: Minimum area threshold
        
    Returns:
        Bounding box [x, y, width, height] or None if mask is too small
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < min_area:
        return None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    return [float(x), float(y), float(w), float(h)]


def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes.
    
    Args:
        bbox1: First box [x, y, w, h] or [x1, y1, x2, y2]
        bbox2: Second box [x, y, w, h] or [x1, y1, x2, y2]
        
    Returns:
        IoU score (0 to 1)
    """
    # Convert to [x1, y1, x2, y2] format if needed
    if len(bbox1) == 4:
        if bbox1[2] < bbox1[0]:  # Already in x1,y1,x2,y2 format
            x1_1, y1_1, x2_1, y2_1 = bbox1
        else:  # x, y, w, h format
            x1_1, y1_1 = bbox1[0], bbox1[1]
            x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    
    if len(bbox2) == 4:
        if bbox2[2] < bbox2[0]:
            x1_2, y1_2, x2_2, y2_2 = bbox2
        else:
            x1_2, y1_2 = bbox2[0], bbox2[1]
            x2_2, y2_2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def filter_overlapping_bboxes(
    bboxes: List[List[float]],
    masks: List[np.ndarray],
    iou_threshold: float = 0.7
) -> Tuple[List[List[float]], List[np.ndarray]]:
    """
    Filter overlapping bounding boxes, keeping the one with larger mask area.
    
    Args:
        bboxes: List of bounding boxes
        masks: Corresponding masks
        iou_threshold: IoU threshold for considering overlap
        
    Returns:
        Filtered bboxes and masks
    """
    if len(bboxes) <= 1:
        return bboxes, masks
    
    # Calculate areas
    areas = [np.sum(m) for m in masks]
    
    # Sort by area descending
    indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    
    keep = []
    for i in indices:
        should_keep = True
        for j in keep:
            iou = calculate_bbox_iou(bboxes[i], bboxes[j])
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(i)
    
    filtered_bboxes = [bboxes[i] for i in keep]
    filtered_masks = [masks[i] for i in keep]
    
    return filtered_bboxes, filtered_masks
