#!/usr/bin/env python3
"""
Dataset Generator Module

Generates COCO-format datasets from PerSAM-F detection results.
Outputs masks, bounding boxes, and coordinate annotations.

Usage:
    from src.inference.dataset_generator import DatasetGenerator
    
    generator = DatasetGenerator()
    generator.generate_dataset(results, output_dir)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
from datetime import datetime

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import utilities
try:
    from src.utils.coco_utils import (
        create_coco_annotation,
        create_coco_image_entry,
        create_coco_dataset
    )
    from src.utils.mask_utils import mask_to_rle, mask_to_polygon
    from src.utils.bbox_utils import get_bbox_from_mask
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


class DatasetGenerator:
    """
    Generate labeled datasets from PerSAM-F detection results.
    
    Outputs:
        - Pixel-level segmentation masks
        - Bounding boxes in multiple formats
        - COCO-format annotations
        - JSON with detection coordinates
    
    Attributes:
        categories: List of category definitions
        output_format: Annotation format ('coco', 'yolo', 'json')
    """
    
    DEFAULT_CATEGORIES = [
        {'id': 1, 'name': 'stone', 'supercategory': 'contaminant'}
    ]
    
    def __init__(
        self,
        categories: Optional[List[Dict]] = None,
        output_format: str = 'coco',
        save_masks: bool = True,
        save_overlays: bool = True,
        overlay_alpha: float = 0.5,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
    ):
        """
        Initialize dataset generator.
        
        Args:
            categories: Category definitions for annotations
            output_format: Output format ('coco', 'yolo', 'json')
            save_masks: Whether to save individual mask images
            save_overlays: Whether to save overlay visualizations
            overlay_alpha: Alpha for overlay blending
            overlay_color: RGB color for mask overlay
        """
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.output_format = output_format
        self.save_masks = save_masks
        self.save_overlays = save_overlays
        self.overlay_alpha = overlay_alpha
        self.overlay_color = overlay_color
    
    def generate_dataset(
        self,
        inference_results: List[Dict],
        output_dir: Union[str, Path],
        image_source_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Generate complete dataset from inference results.
        
        Args:
            inference_results: List of results from PerSAMInference
            output_dir: Directory to save all outputs
            image_source_dir: Optional source directory for images
            
        Returns:
            Dataset statistics dictionary
        """
        output_dir = Path(output_dir)
        
        # Create output directories
        dirs = self._create_output_dirs(output_dir)
        
        # Initialize COCO dataset
        coco_dataset = {
            'info': {
                'description': 'PerSAM-F Stone Detection Dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat(),
            },
            'licenses': [],
            'categories': self.categories,
            'images': [],
            'annotations': [],
        }
        
        # Process each image result
        annotation_id = 1
        stats = {
            'total_images': 0,
            'images_with_detections': 0,
            'total_detections': 0,
            'detection_per_image': [],
        }
        
        for img_idx, result in enumerate(inference_results):
            if 'error' in result:
                logger.warning(f"Skipping {result.get('source', 'unknown')}: {result['error']}")
                continue
            
            source_path = result.get('source')
            if not source_path:
                logger.warning(f"Result {img_idx} has no source path, skipping")
                continue
            
            source_path = Path(source_path)
            detections = result.get('detections', [])
            image_size = result.get('image_size', (0, 0))
            
            stats['total_images'] += 1
            stats['total_detections'] += len(detections)
            stats['detection_per_image'].append(len(detections))
            
            if detections:
                stats['images_with_detections'] += 1
            
            # Add COCO image entry
            image_id = img_idx + 1
            coco_dataset['images'].append({
                'id': image_id,
                'file_name': source_path.name,
                'width': image_size[1],
                'height': image_size[0],
            })
            
            # Load original image for overlays
            if self.save_overlays and source_path.exists():
                original_image = cv2.imread(str(source_path))
                overlay_image = original_image.copy()
            else:
                original_image = None
                overlay_image = None
            
            # Process each detection
            for det_idx, detection in enumerate(detections):
                mask = detection.get('mask')
                bbox = detection.get('bbox')
                score = detection.get('score', 1.0)
                area = detection.get('area', 0)
                centroid = detection.get('centroid', (0, 0))
                
                if mask is None:
                    continue
                
                # Create COCO annotation
                annotation = self._create_annotation(
                    annotation_id=annotation_id,
                    image_id=image_id,
                    mask=mask,
                    bbox=bbox,
                    area=area,
                    score=score,
                )
                coco_dataset['annotations'].append(annotation)
                
                # Save individual mask
                if self.save_masks and mask is not None:
                    mask_filename = f"{source_path.stem}_mask_{det_idx}.png"
                    mask_path = dirs['masks'] / mask_filename
                    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                
                # Add to overlay
                if overlay_image is not None and mask is not None:
                    overlay_image = self._add_mask_to_overlay(
                        overlay_image, mask, self.overlay_color
                    )
                
                annotation_id += 1
            
            # Save overlay image
            if self.save_overlays and overlay_image is not None:
                overlay_path = dirs['overlays'] / f"{source_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay_image)
        
        # Save COCO annotations
        coco_path = output_dir / 'annotations.json'
        with open(coco_path, 'w') as f:
            json.dump(coco_dataset, f, indent=2)
        logger.info(f"Saved COCO annotations: {coco_path}")
        
        # Save detection JSON (simpler format)
        detections_json = self._create_detections_json(inference_results)
        detections_path = output_dir / 'detections.json'
        with open(detections_path, 'w') as f:
            json.dump(detections_json, f, indent=2)
        logger.info(f"Saved detections JSON: {detections_path}")
        
        # Compute and save statistics
        if stats['total_images'] > 0:
            stats['avg_detections_per_image'] = (
                stats['total_detections'] / stats['total_images']
            )
        else:
            stats['avg_detections_per_image'] = 0
        
        stats_path = output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics: {stats_path}")
        
        logger.info(f"Dataset generation complete:")
        logger.info(f"  Images: {stats['total_images']}")
        logger.info(f"  Detections: {stats['total_detections']}")
        logger.info(f"  Avg detections/image: {stats['avg_detections_per_image']:.2f}")
        
        return stats
    
    def _create_output_dirs(self, output_dir: Path) -> Dict[str, Path]:
        """Create all output directories."""
        dirs = {
            'root': output_dir,
            'masks': output_dir / 'masks',
            'overlays': output_dir / 'visualizations',
            'annotations': output_dir / 'annotations',
        }
        
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    def _create_annotation(
        self,
        annotation_id: int,
        image_id: int,
        mask: np.ndarray,
        bbox: List[int],
        area: int,
        score: float,
    ) -> Dict:
        """Create a COCO-format annotation."""
        # Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = bbox
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        
        # Create segmentation (polygon or RLE)
        segmentation = self._mask_to_segmentation(mask)
        
        return {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 1,  # stone
            'bbox': bbox_xywh,
            'area': int(area),
            'segmentation': segmentation,
            'iscrowd': 0,
            'score': float(score),
        }
    
    def _mask_to_segmentation(self, mask: np.ndarray) -> List:
        """Convert binary mask to polygon segmentation."""
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        segmentation = []
        for contour in contours:
            if len(contour) >= 3:
                # Flatten contour to list of coordinates
                polygon = contour.flatten().tolist()
                if len(polygon) >= 6:  # minimum 3 points
                    segmentation.append(polygon)
        
        return segmentation
    
    def _add_mask_to_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Add colored mask overlay to image."""
        # Create color overlay
        colored_mask = np.zeros_like(image)
        # OpenCV uses BGR
        colored_mask[mask > 0] = [color[2], color[1], color[0]]
        
        # Blend
        image = cv2.addWeighted(
            image, 1,
            colored_mask, self.overlay_alpha,
            0
        )
        
        # Draw contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, [color[2], color[1], color[0]], 2)
        
        return image
    
    def _create_detections_json(
        self,
        results: List[Dict],
    ) -> Dict:
        """Create simplified detections JSON with coordinates."""
        output = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'images': [],
        }
        
        for result in results:
            if 'error' in result:
                continue
            
            source = result.get('source', 'unknown')
            detections = []
            
            for det in result.get('detections', []):
                box = det.get('bbox', [0, 0, 0, 0])
                centroid = det.get('centroid', (0, 0))
                
                detections.append({
                    'bbox': box,
                    'center_x': centroid[0],
                    'center_y': centroid[1],
                    'area': det.get('area', 0),
                    'confidence': det.get('score', 0),
                })
            
            output['images'].append({
                'filename': Path(source).name if source else 'unknown',
                'path': source,
                'detection_count': len(detections),
                'detections': detections,
            })
        
        return output


def generate_dataset_from_images(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    persam_weights: Optional[str] = None,
    sam_type: str = 'vit_h',
    **kwargs,
) -> Dict:
    """
    Full pipeline: inference + dataset generation.
    
    Args:
        image_dir: Directory with input images
        output_dir: Output directory for dataset
        persam_weights: Optional PerSAM-F weights
        sam_type: SAM variant
        **kwargs: Additional arguments
        
    Returns:
        Dataset statistics
    """
    from src.inference.persam_inference import PerSAMInference
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    
    # Collect images
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [p for p in image_dir.iterdir() 
              if p.suffix.lower() in extensions]
    
    if not images:
        logger.warning(f"No images found in {image_dir}")
        return {'error': 'No images found'}
    
    # Run inference
    inference = PerSAMInference(
        persam_weights=persam_weights,
        sam_type=sam_type,
        **kwargs,
    )
    results = inference.process_batch(images)
    
    # Generate dataset
    generator = DatasetGenerator()
    stats = generator.generate_dataset(results, output_dir)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dataset from images")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output dataset directory")
    parser.add_argument("--weights", type=str, default=None,
                        help="PerSAM-F weights path")
    parser.add_argument("--sam-type", type=str, default="vit_h",
                        choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
                        help="SAM model variant")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Generation")
    print("=" * 60)
    
    stats = generate_dataset_from_images(
        image_dir=args.input,
        output_dir=args.output,
        persam_weights=args.weights,
        sam_type=args.sam_type,
    )
    
    print("\nDataset Statistics:")
    print(f"  Total images: {stats.get('total_images', 0)}")
    print(f"  Images with detections: {stats.get('images_with_detections', 0)}")
    print(f"  Total detections: {stats.get('total_detections', 0)}")
    print(f"\nDataset saved to: {args.output}")
