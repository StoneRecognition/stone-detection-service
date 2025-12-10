"""
Visualization Utilities Module

Utilities for visualization and overlay creation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from .mask_utils import create_mask_visualization


# Default color mapping for different detection stages
DEFAULT_STAGE_COLORS = {
    'yolo': (0, 255, 0),       # Green for YOLO
    'sam_only': (255, 0, 0),   # Red for SAM only
    'enhanced': (255, 255, 0), # Cyan for enhanced
    'yolo+sam': (255, 255, 0), # Cyan for YOLO+SAM combined
    'unknown': (255, 255, 255) # White for unknown
}


def create_overlay_from_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    contour_thickness: int = 2,
    seed: int = 42
) -> np.ndarray:
    """
    Create an overlay visualization from masks drawn on an image.
    
    Args:
        image: Input image (RGB or BGR)
        masks: List of binary masks
        contour_thickness: Thickness of contour lines
        seed: Random seed for consistent colors
        
    Returns:
        Image with contour overlay
    """
    overlay = image.copy()
    rng = np.random.default_rng(seed)
    
    for mask in masks:
        seg = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(x) for x in rng.integers(0, 255, size=3))
        cv2.drawContours(overlay, contours, -1, color, contour_thickness)
    
    return overlay


def save_overlay(
    image: np.ndarray,
    masks: List[np.ndarray],
    output_path: Union[str, Path],
    is_rgb: bool = True,
    contour_thickness: int = 2,
    seed: int = 42
) -> None:
    """
    Create and save an overlay visualization.
    
    Args:
        image: Input image
        masks: List of binary masks
        output_path: Path to save the overlay
        is_rgb: If True, convert from RGB to BGR before saving
        contour_thickness: Thickness of contour lines
        seed: Random seed for consistent colors
    """
    overlay = create_overlay_from_masks(image, masks, contour_thickness, seed)
    if is_rgb:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), overlay)


def save_inference_results(
    image: np.ndarray,
    combined_mask: np.ndarray,
    output_name: str,
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    kernel_size: int = 5
) -> Tuple[str, str]:
    """
    Save inference results (original image and processed mask).
    
    Args:
        image: Original image (BGR)
        combined_mask: Combined binary mask
        output_name: Base name for output files
        images_dir: Directory to save images
        masks_dir: Directory to save masks
        kernel_size: Morphological kernel size
        
    Returns:
        Tuple of (image_path, mask_path)
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Process and save mask
    mask = create_mask_visualization(combined_mask, kernel_size)
    mask_path = masks_dir / f"{output_name}_mask.png"
    cv2.imwrite(str(mask_path), mask)
    
    # Save original image
    image_path = images_dir / f"{output_name}.png"
    cv2.imwrite(str(image_path), image)
    
    return str(image_path), str(mask_path)


def draw_detections_on_image(
    image: np.ndarray,
    detections: List[Dict],
    stage_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 2,
    show_labels: bool = True,
    label_format: str = "{class_name}: {confidence:.2f} ({stage})"
) -> np.ndarray:
    """
    Draw detection bounding boxes on an image with stage-based coloring.
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries with 'bbox', 'confidence', 
                   'class_name', and optional 'stage' keys
        stage_colors: Optional color mapping for stages (BGR tuples).
                     Defaults to DEFAULT_STAGE_COLORS.
        line_thickness: Thickness of bounding box lines
        font_scale: Scale of label text
        font_thickness: Thickness of label text
        show_labels: Whether to draw labels above boxes
        label_format: Format string for labels. Available placeholders:
                     {class_name}, {confidence}, {stage}
    
    Returns:
        Image with drawn detections
    """
    vis_image = image.copy()
    colors = stage_colors or DEFAULT_STAGE_COLORS
    
    for detection in detections:
        bbox = detection.get('bbox', [0, 0, 0, 0])
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class_name', 'object')
        stage_type = detection.get('stage', 'unknown')
        
        # Get color for this stage
        color = colors.get(stage_type, colors.get('unknown', (255, 255, 255)))
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        if show_labels:
            label = label_format.format(
                class_name=class_name,
                confidence=confidence,
                stage=stage_type
            )
            cv2.putText(
                vis_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness
            )
    
    return vis_image
