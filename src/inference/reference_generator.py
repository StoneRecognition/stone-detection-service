#!/usr/bin/env python3
"""
Automated Reference Mask Generator

Generates a single high-quality reference segmentation mask automatically
using Grounding DINO for detection and SAM for precise segmentation.

This is Step 1.1 of the PerSAM-F pipeline - producing the "one-shot" 
reference mask required for fine-tuning.

Usage:
    from src.inference.reference_generator import ReferenceGenerator
    
    generator = ReferenceGenerator()
    ref_image, ref_mask = generator.generate(
        image_path="data/raw/sample.jpg",
        text_prompt="stone contaminant"
    )
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging

import cv2
import numpy as np
import torch

# Add project root to path for imports
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
        'text_prompt': config.get('persam.text_prompt', 'stone contaminant'),
        'confidence_threshold': config.get('persam.confidence_threshold', 0.3),
        'sam_type': config.get('persam.sam_type', 'vit_h'),
    }
except ImportError:
    weights_dir = Path('./weight')
    persam_config = {
        'text_prompt': 'stone contaminant',
        'confidence_threshold': 0.3,
        'sam_type': 'vit_h',
    }


class ReferenceGenerator:
    """
    Automated reference mask generator for PerSAM-F.
    
    Combines Grounding DINO (text-prompt detection) with SAM 
    (precise segmentation) to automatically generate a high-quality
    reference mask from a single image.
    
    Attributes:
        detector: Grounding DINO detector instance
        sam_predictor: SAM predictor instance
        text_prompt: Default text prompt for detection
        confidence_threshold: Minimum confidence for detection
    """
    
    # SAM model variants
    SAM_TYPES = {
        'vit_h': 'sam_vit_h.pt',
        'vit_l': 'sam_vit_l.pt', 
        'vit_b': 'sam_vit_b.pt',
        'vit_t': 'mobile_sam.pt',  # MobileSAM
    }
    
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        sam_type: str = 'vit_h',
        grounding_dino_weights: Optional[str] = None,
        text_prompt: str = "stone contaminant",
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
    ):
        """
        Initialize the reference generator.
        
        Args:
            sam_checkpoint: Path to SAM weights. Auto-detected if None.
            sam_type: SAM variant ('vit_h', 'vit_l', 'vit_b', 'vit_t')
            grounding_dino_weights: Path to Grounding DINO weights.
            text_prompt: Default text prompt for object detection.
            confidence_threshold: Minimum detection confidence.
            device: Computation device ('cuda', 'cpu', or None for auto).
        """
        self.text_prompt = text_prompt
        self.confidence_threshold = confidence_threshold
        self.sam_type = sam_type
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Set SAM checkpoint path
        if sam_checkpoint is None:
            sam_weights_file = self.SAM_TYPES.get(sam_type, 'sam_vit_h.pt')
            # Check both 'weights' and 'weight' directories
            if (weights_dir / sam_weights_file).exists():
                sam_checkpoint = str(weights_dir / sam_weights_file)
            elif (project_root / 'weight' / sam_weights_file).exists():
                sam_checkpoint = str(project_root / 'weight' / sam_weights_file)
            else:
                raise FileNotFoundError(
                    f"SAM weights not found. Expected: {weights_dir / sam_weights_file}"
                )
        
        self.sam_checkpoint = sam_checkpoint
        self.grounding_dino_weights = grounding_dino_weights
        
        # Lazy load models
        self._detector = None
        self._sam_predictor = None
    
    @property
    def detector(self):
        """Lazy load Grounding DINO detector."""
        if self._detector is None:
            from src.inference.grounding_dino import GroundingDINODetector
            self._detector = GroundingDINODetector(
                weights_path=self.grounding_dino_weights,
                device=self.device,
                box_threshold=self.confidence_threshold,
            )
            logger.info("Grounding DINO detector loaded")
        return self._detector
    
    @property
    def sam_predictor(self):
        """Lazy load SAM predictor."""
        if self._sam_predictor is None:
            self._sam_predictor = self._load_sam_predictor()
            logger.info("SAM predictor loaded")
        return self._sam_predictor
    
    def _load_sam_predictor(self):
        """Load the appropriate SAM predictor based on model type."""
        if self.sam_type == 'vit_t':
            # MobileSAM
            from mobile_sam import sam_model_registry, SamPredictor
            model_type = 'vit_t'
        else:
            # Standard SAM
            from segment_anything import sam_model_registry, SamPredictor
            model_type = self.sam_type
        
        sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        return SamPredictor(sam)
    
    def detect_target(
        self,
        image: Union[np.ndarray, str, Path],
        text_prompt: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Detect the target object using Grounding DINO.
        
        Args:
            image: Input image (numpy array or path)
            text_prompt: Override default text prompt
            
        Returns:
            Highest confidence detection or None
        """
        prompt = text_prompt or self.text_prompt
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image_array = cv2.imread(str(image))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_array = image
        
        # Detect with Grounding DINO
        detections = self.detector.detect_with_absolute_boxes(prompt, image_array)
        
        if not detections["confidences"]:
            logger.warning(f"No objects detected with prompt: '{prompt}'")
            return None
        
        # Get highest confidence detection
        best = self.detector.select_highest_confidence(detections)
        logger.info(
            f"Best detection: {best['label']} with confidence {best['confidence']:.3f}"
        )
        
        return best
    
    def generate_mask_from_box(
        self,
        image: np.ndarray,
        box: List[int],
    ) -> Tuple[np.ndarray, float]:
        """
        Generate segmentation mask using SAM with box prompt.
        
        Args:
            image: RGB image array
            box: Bounding box [x1, y1, x2, y2] in pixels
            
        Returns:
            Tuple of (mask, score) where mask is binary numpy array
        """
        # Set image in predictor
        self.sam_predictor.set_image(image)
        
        # Convert box to numpy array
        input_box = np.array(box)
        
        # Generate masks
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # Add batch dimension
            multimask_output=True,
        )
        
        # Select best mask (highest score)
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        logger.info(f"Generated mask with score: {best_score:.3f}")
        
        return best_mask, best_score
    
    def generate_mask_from_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        point_label: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate segmentation mask using SAM with point prompt.
        
        Args:
            image: RGB image array
            point: (x, y) center point in pixels
            point_label: 1 for foreground, 0 for background
            
        Returns:
            Tuple of (mask, score)
        """
        # Set image in predictor
        self.sam_predictor.set_image(image)
        
        # Prepare point input
        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([point_label])
        
        # Generate masks
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def generate(
        self,
        image: Union[np.ndarray, str, Path],
        text_prompt: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        use_point_prompt: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate reference mask automatically.
        
        This is the main entry point that combines:
        1. Grounding DINO detection
        2. SAM mask generation
        
        Args:
            image: Input image (numpy array or path)
            text_prompt: Override default text prompt
            output_dir: Optional directory to save outputs
            use_point_prompt: Use center point instead of box prompt
            
        Returns:
            Tuple of (image, mask, metadata) where:
                - image: RGB numpy array
                - mask: Binary mask numpy array
                - metadata: Dict with detection info
        """
        # Load image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image_array = cv2.imread(str(image_path))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_path = None
            image_array = image.copy()
        
        # Step 1: Detect target with Grounding DINO
        detection = self.detect_target(image_array, text_prompt)
        
        if detection is None:
            raise ValueError(
                f"No objects detected with prompt: '{text_prompt or self.text_prompt}'. "
                "Try a different prompt or lower confidence threshold."
            )
        
        # Step 2: Generate mask with SAM
        if use_point_prompt:
            # Use center point of bounding box
            center = self.detector.get_center_point(
                detection['box'], 
                detection['image_size']
            )
            mask, score = self.generate_mask_from_point(image_array, center)
            prompt_type = "point"
        else:
            # Use bounding box
            box = detection['box_absolute']
            mask, score = self.generate_mask_from_box(image_array, box)
            prompt_type = "box"
        
        # Prepare metadata
        metadata = {
            'detection': {
                'label': detection['label'],
                'confidence': float(detection['confidence']),
                'box': detection['box'],
                'box_absolute': detection.get('box_absolute'),
            },
            'segmentation': {
                'score': float(score),
                'prompt_type': prompt_type,
                'mask_area': int(mask.sum()),
            },
            'text_prompt': text_prompt or self.text_prompt,
            'sam_type': self.sam_type,
        }
        
        if image_path:
            metadata['source_image'] = str(image_path)
        
        # Save outputs if directory specified
        if output_dir is not None:
            self._save_outputs(image_array, mask, metadata, output_dir, image_path)
        
        return image_array, mask, metadata
    
    def _save_outputs(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        metadata: Dict,
        output_dir: Union[str, Path],
        source_path: Optional[Path] = None,
    ):
        """Save reference image, mask, and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate base name
        if source_path:
            base_name = source_path.stem
        else:
            base_name = "reference"
        
        # Save reference image
        image_path = output_dir / f"{base_name}_reference.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved reference image: {image_path}")
        
        # Save mask
        mask_path = output_dir / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        logger.info(f"Saved reference mask: {mask_path}")
        
        # Save overlay visualization
        overlay = self._create_overlay(image, mask)
        overlay_path = output_dir / f"{base_name}_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved overlay: {overlay_path}")
        
        # Save metadata
        metadata_path = output_dir / f"{base_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
    
    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create visualization overlay of mask on image."""
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        # Draw contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        return overlay


def generate_reference(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    text_prompt: str = "stone contaminant",
    sam_type: str = "vit_h",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience function to generate reference mask.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        text_prompt: Text prompt for detection
        sam_type: SAM variant to use
        **kwargs: Additional arguments for ReferenceGenerator
        
    Returns:
        Tuple of (image, mask, metadata)
    """
    generator = ReferenceGenerator(
        text_prompt=text_prompt,
        sam_type=sam_type,
        **kwargs
    )
    return generator.generate(image_path, output_dir=output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate reference mask for PerSAM-F"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="results/reference",
                        help="Output directory")
    parser.add_argument("--prompt", type=str, default="stone contaminant",
                        help="Text prompt for detection")
    parser.add_argument("--sam-type", type=str, default="vit_h",
                        choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
                        help="SAM model variant")
    parser.add_argument("--use-point", action="store_true",
                        help="Use point prompt instead of box")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    print(f"Generating reference mask for: {args.image}")
    print(f"Text prompt: '{args.prompt}'")
    print(f"SAM type: {args.sam_type}")
    
    generator = ReferenceGenerator(
        text_prompt=args.prompt,
        sam_type=args.sam_type,
        confidence_threshold=args.confidence,
    )
    
    image, mask, metadata = generator.generate(
        args.image,
        output_dir=args.output,
        use_point_prompt=args.use_point,
    )
    
    print(f"\nResults saved to: {args.output}")
    print(f"Detection confidence: {metadata['detection']['confidence']:.3f}")
    print(f"Segmentation score: {metadata['segmentation']['score']:.3f}")
    print(f"Mask area: {metadata['segmentation']['mask_area']} pixels")
