"""
Mask Processing Utilities Module

Utilities for mask processing, post-processing, and compression.
"""

import cv2
import zlib
import base64
import numpy as np
from typing import Dict, Any, Optional


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


def compress_mask_rle(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """Compress a binary mask using run-length encoding and zlib.
    
    Args:
        mask: Binary mask (2D numpy array)
        
    Returns:
        Dictionary with compressed mask data, or None if compression fails
    """
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
    except Exception:
        return None


def decompress_mask_rle(compressed: Dict[str, Any]) -> Optional[np.ndarray]:
    """Decompress a run-length encoded mask.
    
    Args:
        compressed: Compressed mask dictionary from compress_mask_rle
        
    Returns:
        Reconstructed binary mask, or None if decompression fails
    """
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
    except Exception:
        return None


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
