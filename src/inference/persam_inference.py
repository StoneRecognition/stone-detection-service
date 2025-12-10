#!/usr/bin/env python3
"""
PerSAM-F Inference Module

Batch inference using fine-tuned PerSAM-F model for stone detection.
Implements "segment everything" mode with dense grid point prompting.

Usage:
    from src.inference.persam_inference import PerSAMInference
    
    inference = PerSAMInference("weights/persam_stone.pth")
    results = inference.process_batch(image_paths)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load config
try:
    from src.utils.settings import config
    weights_dir = Path(config.get('paths.weights_dir', 'weight'))
    persam_config = {
        'points_per_side': config.get('persam.inference.points_per_side', 32),
        'iou_threshold': config.get('persam.inference.iou_threshold', 0.7),
        'stability_threshold': config.get('persam.inference.stability_threshold', 0.9),
        'min_mask_area': config.get('persam.inference.min_mask_area', 100),
    }
except ImportError:
    weights_dir = Path('./weights')
    persam_config = {
        'points_per_side': 32,
        'iou_threshold': 0.7,
        'stability_threshold': 0.9,
        'min_mask_area': 100,
    }


class PerSAMInference:
    """
    PerSAM-F inference for batch stone detection.
    
    Uses the fine-tuned mask weights to produce specialized
    segmentation for the target stone category.
    
    Attributes:
        model: SAM model with loaded weights
        mask_weights: Trained aggregation weights
        points_per_side: Grid density for automatic segmentation
        iou_threshold: Minimum predicted IoU for valid masks
    """
    
    SAM_TYPES = {
        'vit_h': 'sam_vit_h.pt',
        'vit_l': 'sam_vit_l.pt',
        'vit_b': 'sam_vit_b.pt',
        'vit_t': 'mobile_sam.pt',
    }
    
    def __init__(
        self,
        persam_weights: Optional[Union[str, Path]] = None,
        sam_checkpoint: Optional[str] = None,
        sam_type: str = 'vit_h',
        device: Optional[str] = None,
        points_per_side: int = 32,
        iou_threshold: float = 0.7,
        stability_threshold: float = 0.9,
        min_mask_area: int = 100,
    ):
        """
        Initialize PerSAM-F inference.
        
        Args:
            persam_weights: Path to trained PerSAM-F weights
            sam_checkpoint: Path to base SAM weights
            sam_type: SAM variant
            device: Computation device
            points_per_side: Grid points per side for automatic mode
            iou_threshold: Minimum IoU for valid detections
            stability_threshold: Mask stability threshold
            min_mask_area: Minimum mask area in pixels
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.sam_type = sam_type
        self.points_per_side = points_per_side
        self.iou_threshold = iou_threshold
        self.stability_threshold = stability_threshold
        self.min_mask_area = min_mask_area
        
        # Load PerSAM-F weights if provided
        self.mask_weights = None
        if persam_weights and Path(persam_weights).exists():
            checkpoint = torch.load(persam_weights, map_location=self.device)
            self.mask_weights = checkpoint['mask_weights'].to(self.device)
            self.sam_type = checkpoint.get('sam_type', sam_type)
            logger.info(f"Loaded PerSAM-F weights from: {persam_weights}")
        
        # Set SAM checkpoint
        if sam_checkpoint is None:
            sam_file = self.SAM_TYPES.get(self.sam_type, 'sam_vit_h.pt')
            for check_dir in [weights_dir, project_root / 'weight']:
                check_path = check_dir / sam_file
                if check_path.exists():
                    sam_checkpoint = str(check_path)
                    break
        
        self.sam_checkpoint = sam_checkpoint
        
        # Load model and predictor
        self.model = self._load_model()
        self.predictor = self._create_predictor()
        self.mask_generator = self._create_mask_generator()
        
        logger.info(f"PerSAM-F Inference initialized on {self.device}")
    
    def _load_model(self):
        """Load SAM model."""
        if self.sam_type == 'vit_t':
            from mobile_sam import sam_model_registry
            model_type = 'vit_t'
        else:
            from segment_anything import sam_model_registry
            model_type = self.sam_type
        
        model = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        model.to(device=self.device)
        model.eval()
        return model
    
    def _create_predictor(self):
        """Create SAM predictor."""
        if self.sam_type == 'vit_t':
            from mobile_sam import SamPredictor
        else:
            from segment_anything import SamPredictor
        return SamPredictor(self.model)
    
    def _create_mask_generator(self):
        """Create automatic mask generator."""
        if self.sam_type == 'vit_t':
            from mobile_sam import SamAutomaticMaskGenerator
        else:
            from segment_anything import SamAutomaticMaskGenerator
        
        return SamAutomaticMaskGenerator(
            self.model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.iou_threshold,
            stability_score_thresh=self.stability_threshold,
            min_mask_region_area=self.min_mask_area,
        )
    
    def _apply_learned_weights(
        self,
        masks: List[np.ndarray],
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Apply learned mask weights for aggregation.
        
        When PerSAM-F weights are available, uses them to select
        the optimal scale mask.
        """
        if self.mask_weights is None or len(masks) < 3:
            # No trained weights, use highest score
            best_idx = np.argmax(scores)
            return masks[best_idx], scores[best_idx]
        
        # Apply softmax to weights
        weights = F.softmax(self.mask_weights, dim=0).cpu().numpy()
        
        # Weighted aggregation of masks
        aggregated = np.zeros_like(masks[0], dtype=np.float32)
        for i, (mask, weight) in enumerate(zip(masks[:3], weights)):
            aggregated += weight * mask.astype(np.float32)
        
        # Threshold to binary
        binary_mask = (aggregated > 0.5).astype(np.uint8)
        
        # Compute weighted score
        weighted_score = sum(w * s for w, s in zip(weights, scores[:3]))
        
        return binary_mask, float(weighted_score)
    
    def segment_with_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        point_label: int = 1,
    ) -> Dict:
        """
        Segment using a point prompt.
        
        Args:
            image: RGB image array
            point: (x, y) point coordinates
            point_label: 1 for foreground, 0 for background
            
        Returns:
            Dict with mask, score, and metadata
        """
        self.predictor.set_image(image)
        
        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([point_label])
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Apply learned weights if available
        best_mask, best_score = self._apply_learned_weights(masks, scores)
        
        return {
            'mask': best_mask,
            'score': best_score,
            'point': point,
        }
    
    def segment_with_box(
        self,
        image: np.ndarray,
        box: List[int],
    ) -> Dict:
        """
        Segment using a bounding box prompt.
        
        Args:
            image: RGB image array
            box: [x1, y1, x2, y2] bounding box
            
        Returns:
            Dict with mask, score, and metadata
        """
        self.predictor.set_image(image)
        
        input_box = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        best_mask, best_score = self._apply_learned_weights(masks, scores)
        
        return {
            'mask': best_mask,
            'score': best_score,
            'box': box,
        }
    
    def segment_everything(
        self,
        image: np.ndarray,
    ) -> List[Dict]:
        """
        Segment all objects in image using automatic mode.
        
        Uses dense grid point prompting to find all potential
        stone instances in the image.
        
        Args:
            image: RGB image array
            
        Returns:
            List of detection dicts with mask, bbox, score, area
        """
        # Generate all masks
        masks = self.mask_generator.generate(image)
        
        # Process each mask
        detections = []
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            
            # Skip small masks
            if area < self.min_mask_area:
                continue
            
            # Get bounding box
            bbox = mask_data['bbox']  # [x, y, w, h] format
            x, y, w, h = bbox
            box_xyxy = [x, y, x + w, y + h]
            
            # Get centroid
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                centroid = (int(x_indices.mean()), int(y_indices.mean()))
            else:
                centroid = (x + w // 2, y + h // 2)
            
            detections.append({
                'mask': mask.astype(np.uint8),
                'bbox': box_xyxy,
                'bbox_xywh': bbox,
                'area': area,
                'centroid': centroid,
                'score': mask_data['predicted_iou'],
                'stability_score': mask_data['stability_score'],
            })
        
        # Sort by score
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Found {len(detections)} stone instances")
        return detections
    
    def process_image(
        self,
        image: Union[np.ndarray, str, Path],
        mode: str = 'auto',
    ) -> Dict:
        """
        Process a single image.
        
        Args:
            image: Input image (array or path)
            mode: 'auto' for segment-everything, 'prompt' for single detection
            
        Returns:
            Dict with detections and metadata
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image_array = cv2.imread(str(image_path))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_path = None
            image_array = image
        
        height, width = image_array.shape[:2]
        
        # Run detection
        if mode == 'auto':
            detections = self.segment_everything(image_array)
        else:
            # Use center point as prompt
            center = (width // 2, height // 2)
            result = self.segment_with_point(image_array, center)
            detections = [result] if result['score'] > self.iou_threshold else []
        
        return {
            'detections': detections,
            'image_size': (height, width),
            'source': str(image_path) if image_path else None,
            'detection_count': len(detections),
        }
    
    def process_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        mode: str = 'auto',
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Process a batch of images.
        
        Args:
            images: List of images (paths or arrays)
            mode: Detection mode ('auto' or 'prompt')
            show_progress: Show progress bar
            
        Returns:
            List of results for each image
        """
        results = []
        
        iterator = tqdm(images, desc="Processing images") if show_progress else images
        
        for image in iterator:
            try:
                result = self.process_image(image, mode=mode)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image}: {e}")
                results.append({
                    'detections': [],
                    'error': str(e),
                    'source': str(image) if not isinstance(image, np.ndarray) else None,
                })
        
        total_detections = sum(r.get('detection_count', 0) for r in results)
        logger.info(f"Processed {len(results)} images, found {total_detections} total detections")
        
        return results
    
    def filter_by_confidence(
        self,
        detections: List[Dict],
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of detection dicts
            threshold: Confidence threshold (uses self.iou_threshold if None)
            
        Returns:
            Filtered list of detections
        """
        threshold = threshold or self.iou_threshold
        return [d for d in detections if d.get('score', 0) >= threshold]


def run_inference(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    persam_weights: Optional[str] = None,
    sam_type: str = 'vit_h',
    **kwargs,
) -> List[Dict]:
    """
    Convenience function to run inference on a directory.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory for output results
        persam_weights: Path to trained PerSAM-F weights
        sam_type: SAM variant
        **kwargs: Additional arguments for PerSAMInference
        
    Returns:
        List of inference results
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image paths
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [p for p in image_dir.iterdir() 
              if p.suffix.lower() in extensions]
    
    if not images:
        logger.warning(f"No images found in {image_dir}")
        return []
    
    logger.info(f"Found {len(images)} images to process")
    
    # Create inference engine
    inference = PerSAMInference(
        persam_weights=persam_weights,
        sam_type=sam_type,
        **kwargs,
    )
    
    # Process all images
    results = inference.process_batch(images)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PerSAM-F batch inference")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image or directory")
    parser.add_argument("--output", "-o", type=str, default="results/persam_inference",
                        help="Output directory")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained PerSAM-F weights")
    parser.add_argument("--sam-type", type=str, default="vit_h",
                        choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
                        help="SAM model variant")
    parser.add_argument("--points-per-side", type=int, default=32,
                        help="Grid points per side")
    parser.add_argument("--iou-threshold", type=float, default=0.7,
                        help="Minimum IoU threshold")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        results = run_inference(
            image_dir=input_path,
            output_dir=args.output,
            persam_weights=args.weights,
            sam_type=args.sam_type,
            points_per_side=args.points_per_side,
            iou_threshold=args.iou_threshold,
        )
        print(f"\nProcessed {len(results)} images")
    else:
        inference = PerSAMInference(
            persam_weights=args.weights,
            sam_type=args.sam_type,
            points_per_side=args.points_per_side,
            iou_threshold=args.iou_threshold,
        )
        result = inference.process_image(input_path)
        print(f"\nFound {result['detection_count']} detections")
