"""
Contour Utilities Module

Utilities for contour analysis and point selection.
"""

import cv2
import numpy as np
from typing import Optional


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
