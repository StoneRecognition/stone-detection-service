"""
Utilities Package

Centralized utilities for the stone-detection-service project.
All functions are organized by single-responsibility modules.
"""

# JSON utilities
from .json_utils import (
    NumpyEncoder,
    convert_numpy_to_json,
    save_json,
    load_json,
)

# COCO annotation utilities
from .coco_utils import (
    load_coco_annotations,
    load_or_create_coco_dataset,
    save_dataset_metadata,
    save_coco_annotations,
    create_coco_annotation_from_mask,
    create_coco_annotations_from_masks,
    create_coco_image_entry,
    build_mask_from_coco,
    draw_coco_overlay,
)

# Mask processing utilities
from .mask_utils import (
    post_process_mask,
    calculate_mask_iou,
    compress_mask_rle,
    decompress_mask_rle,
    create_mask_visualization,
)

# Bounding box utilities
from .bbox_utils import (
    get_bbox_from_mask,
    calculate_bbox_iou,
    filter_overlapping_bboxes,
)

# Visualization utilities
from .visualization_utils import (
    create_overlay_from_masks,
    save_overlay,
    save_inference_results,
    draw_detections_on_image,
    DEFAULT_STAGE_COLORS,
)

# Contour utilities
from .contour_utils import (
    is_closed_contour,
    get_perimeter_points,
    smart_point_selection,
)

# Logging utilities
from .logging_utils import (
    setup_thread_safe_logging,
    stop_logging,
    safe_log,
)

# All public exports
__all__ = [
    # JSON
    'NumpyEncoder',
    'convert_numpy_to_json',
    'save_json',
    'load_json',
    # COCO
    'load_coco_annotations',
    'load_or_create_coco_dataset',
    'save_dataset_metadata',
    'save_coco_annotations',
    'create_coco_annotation_from_mask',
    'create_coco_annotations_from_masks',
    'create_coco_image_entry',
    'build_mask_from_coco',
    'draw_coco_overlay',
    # Masks
    'post_process_mask',
    'calculate_mask_iou',
    'compress_mask_rle',
    'decompress_mask_rle',
    'create_mask_visualization',
    # Bounding boxes
    'get_bbox_from_mask',
    'calculate_bbox_iou',
    'filter_overlapping_bboxes',
    # Visualization
    'create_overlay_from_masks',
    'save_overlay',
    'save_inference_results',
    'draw_detections_on_image',
    'DEFAULT_STAGE_COLORS',
    # Contours
    'is_closed_contour',
    'get_perimeter_points',
    'smart_point_selection',
    # Logging
    'setup_thread_safe_logging',
    'stop_logging',
    'safe_log',
]
