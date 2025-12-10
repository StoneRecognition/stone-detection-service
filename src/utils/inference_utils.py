"""
Inference Utilities Module

Common utility functions used across inference scripts for:
- JSON serialization
- COCO annotation handling
- Mask processing
- Bounding box operations
- Contour analysis
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import gc


# ==============================================================================
# JSON Serialization Utilities
# ==============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy_to_json(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and types to JSON-serializable format.
    
    Args:
        obj: Object to convert (can be dict, list, or numpy type)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with NumPy type handling.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, cls=NumpyEncoder)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==============================================================================
# COCO Annotation Utilities
# ==============================================================================

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


def create_mask_visualization(
    combined_mask: np.ndarray,
    kernel_size: int = 5,
    morph_operation: str = "close"
) -> np.ndarray:
    """
    Create a visualizable mask with morphological processing.
    
    Args:
        combined_mask: Combined binary mask (bool or 0/1)
        kernel_size: Size of morphological kernel
        morph_operation: 'close', 'open', or 'none'
        
    Returns:
        Processed mask ready for saving (uint8, 0-255)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = (combined_mask * 255).astype(np.uint8)
    
    if morph_operation == "close":
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif morph_operation == "open":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, 1)
    
    return mask


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


# ==============================================================================
# Mask Processing Utilities
# ==============================================================================

def post_process_mask(
    mask: np.ndarray,
    min_area: int = 100,
    kernel_size: int = 3,
    apply_blur: bool = True
) -> np.ndarray:
    """
    Post-process a segmentation mask to improve quality.
    
    Args:
        mask: Input binary mask
        min_area: Minimum area for connected components
        kernel_size: Kernel size for morphological operations
        apply_blur: Whether to apply median blur for smoothing
        
    Returns:
        Processed mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Opening to remove small objects
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Closing to fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    
    # Smooth boundaries
    if apply_blur:
        mask = cv2.medianBlur(mask, 3)
    
    return mask.astype(bool)


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score (0 to 1)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


# ==============================================================================
# Bounding Box Utilities
# ==============================================================================

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
    
    filtered_bboxes = []
    filtered_masks = []
    
    for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
        should_keep = True
        
        for j, (existing_bbox, existing_mask) in enumerate(zip(filtered_bboxes, filtered_masks)):
            iou = calculate_bbox_iou(bbox, existing_bbox)
            if iou > iou_threshold:
                # Keep the one with larger mask area
                if np.sum(mask) > np.sum(existing_mask):
                    filtered_bboxes[j] = bbox
                    filtered_masks[j] = mask
                should_keep = False
                break
        
        if should_keep:
            filtered_bboxes.append(bbox)
            filtered_masks.append(mask)
    
    return filtered_bboxes, filtered_masks


# ==============================================================================
# Contour Utilities
# ==============================================================================

def is_closed_contour(contour: np.ndarray, tolerance: float = 2.0) -> bool:
    """
    Check if a contour is closed.
    
    Args:
        contour: OpenCV contour array
        tolerance: Distance tolerance for considering contour closed
        
    Returns:
        True if contour is closed
    """
    if len(contour) < 3:
        return False
    return np.linalg.norm(contour[0][0] - contour[-1][0]) < tolerance


def get_perimeter_points(
    mask: np.ndarray,
    num_points: int = 6
) -> np.ndarray:
    """
    Get evenly spaced points along the perimeter of a mask.
    
    Args:
        mask: Binary mask
        num_points: Number of points to sample
        
    Returns:
        Array of (x, y) points
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )
    
    if not contours:
        return np.array([])
    
    contour = contours[0][:, 0, :]
    if len(contour) < num_points:
        return contour
    
    # Evenly distribute points along perimeter
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
    return contour[indices]


def smart_point_selection(
    mask: np.ndarray,
    num_points: int = 8
) -> np.ndarray:
    """
    Smart selection of representative points from a mask using clustering.
    
    Args:
        mask: Binary mask
        num_points: Number of points to select
        
    Returns:
        Array of (x, y) points
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.array([])
    
    if len(ys) <= num_points:
        return np.column_stack([xs, ys])
    
    # Use clustering for point selection
    try:
        from sklearn.cluster import KMeans
        points = np.column_stack([xs, ys])
        kmeans = KMeans(n_clusters=num_points, random_state=42, n_init=10)
        kmeans.fit(points)
        cluster_centers = kmeans.cluster_centers_.astype(int)
        
        # Add center point
        center_y, center_x = int(np.mean(ys)), int(np.mean(xs))
        center_point = np.array([[center_x, center_y]])
        
        # Add perimeter points
        perimeter_points = get_perimeter_points(mask, num_points // 2)
        
        if len(perimeter_points) > 0:
            all_points = np.vstack([cluster_centers, center_point, perimeter_points])
        else:
            all_points = np.vstack([cluster_centers, center_point])
        
        return all_points
        
    except ImportError:
        # Fallback without sklearn
        return np.column_stack([xs[::len(xs)//num_points], ys[::len(ys)//num_points]])


# ==============================================================================
# Detection Visualization Utilities
# ==============================================================================

# Default color mapping for different detection stages
DEFAULT_STAGE_COLORS = {
    'yolo': (0, 255, 0),       # Green for YOLO
    'sam_only': (255, 0, 0),   # Red for SAM only
    'enhanced': (255, 255, 0), # Cyan for enhanced
    'unknown': (255, 255, 255) # White for unknown
}


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


# ==============================================================================
# Mask Compression Utilities
# ==============================================================================

def compress_mask_rle(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """Compress a binary mask using run-length encoding and zlib.
    
    Args:
        mask: Binary mask (2D numpy array)
        
    Returns:
        Dictionary with compressed mask data, or None if compression fails
    """
    import zlib
    import base64
    
    if mask is None:
        return None
    
    try:
        # Flatten and convert to bytes
        flat = mask.astype(np.uint8).flatten()
        
        # Run-length encode
        rle = []
        current_val = flat[0]
        count = 1
        
        for val in flat[1:]:
            if val == current_val:
                count += 1
            else:
                rle.extend([int(current_val), count])
                current_val = val
                count = 1
        rle.extend([int(current_val), count])
        
        # Compress with zlib
        rle_bytes = np.array(rle, dtype=np.uint32).tobytes()
        compressed = zlib.compress(rle_bytes, level=9)
        
        return {
            'type': 'rle_zlib',
            'shape': list(mask.shape),
            'data': base64.b64encode(compressed).decode('ascii'),
            'original_size': len(flat),
            'compressed_size': len(compressed),
        }
    except Exception as e:
        return None


def decompress_mask_rle(compressed: Dict[str, Any]) -> Optional[np.ndarray]:
    """Decompress a run-length encoded mask.
    
    Args:
        compressed: Compressed mask dictionary from compress_mask_rle
        
    Returns:
        Reconstructed binary mask, or None if decompression fails
    """
    import zlib
    import base64
    
    if compressed is None:
        return None
    
    try:
        # Decode and decompress
        data = base64.b64decode(compressed['data'])
        decompressed = zlib.decompress(data)
        rle = np.frombuffer(decompressed, dtype=np.uint32)
        
        # Reconstruct from RLE
        flat = []
        for i in range(0, len(rle), 2):
            val, count = rle[i], rle[i + 1]
            flat.extend([val] * count)
        
        # Reshape
        shape = tuple(compressed['shape'])
        mask = np.array(flat, dtype=np.uint8).reshape(shape)
        
        return mask
    except Exception as e:
        return None

