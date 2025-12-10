#!/usr/bin/env python3
"""
Grounded-SAM Module

Unified Grounding DINO + SAM integration following the official
Grounded-Segment-Anything repository pattern.

This module provides a clean Python API for the grounded_sam_demo.py functionality,
allowing text-prompt detection and segmentation in a single pipeline.

Usage:
    from src.inference.grounded_sam import GroundedSAM
    
    gsam = GroundedSAM()
    results = gsam.detect_and_segment(
        image_path="data/raw/sample.jpg",
        text_prompt="stone contaminant",
    )

Command-line equivalent:
    python grounded_sam_demo.py \\
      --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \\
      --grounded_checkpoint groundingdino_swint_ogc.pth \\
      --sam_checkpoint sam_vit_h_4b8939.pth \\
      --input_image sample.jpg \\
      --output_dir "outputs" \\
      --box_threshold 0.3 \\
      --text_threshold 0.25 \\
      --text_prompt "stone contaminant" \\
      --device "cuda"

References:
    https://github.com/IDEA-Research/Grounded-Segment-Anything
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import json

import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load config
try:
    from src.utils.settings import config
    WEIGHTS_DIR = Path(config.get('paths.weights_dir', 'weight'))
except ImportError:
    WEIGHTS_DIR = project_root / 'weight'

# Default model paths (relative to weights directory)
DEFAULT_GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_GROUNDING_DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
DEFAULT_SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
DEFAULT_SAM_HQ_CHECKPOINT = "sam_hq_vit_h.pth"


class GroundedSAM:
    """
    Unified Grounded-SAM pipeline combining Grounding DINO detection
    with SAM segmentation in a single interface.
    
    This follows the official Grounded-Segment-Anything implementation
    pattern for maximum compatibility.
    
    Attributes:
        grounding_dino_model: Loaded Grounding DINO model
        sam_predictor: SAM predictor instance
        device: Computation device (cuda/cpu)
        box_threshold: Detection confidence threshold
        text_threshold: Text matching threshold
    """
    
    def __init__(
        self,
        grounding_dino_config: Optional[str] = None,
        grounding_dino_checkpoint: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        sam_hq_checkpoint: Optional[str] = None,
        use_sam_hq: bool = False,
        device: Optional[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """
        Initialize Grounded-SAM pipeline.
        
        Args:
            grounding_dino_config: Path to Grounding DINO config file
            grounding_dino_checkpoint: Path to Grounding DINO weights
            sam_checkpoint: Path to SAM weights (used if use_sam_hq=False)
            sam_hq_checkpoint: Path to SAM-HQ weights (used if use_sam_hq=True)
            use_sam_hq: Whether to use SAM-HQ instead of standard SAM
            device: Computation device ('cuda', 'cpu', or None for auto)
            box_threshold: Detection confidence threshold (default: 0.3)
            text_threshold: Text matching threshold (default: 0.25)
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_sam_hq = use_sam_hq
        
        # Resolve paths
        self.grounding_dino_config = self._resolve_path(
            grounding_dino_config, DEFAULT_GROUNDING_DINO_CONFIG
        )
        self.grounding_dino_checkpoint = self._resolve_path(
            grounding_dino_checkpoint, DEFAULT_GROUNDING_DINO_CHECKPOINT
        )
        
        if use_sam_hq:
            self.sam_checkpoint = self._resolve_path(
                sam_hq_checkpoint, DEFAULT_SAM_HQ_CHECKPOINT
            )
        else:
            self.sam_checkpoint = self._resolve_path(
                sam_checkpoint, DEFAULT_SAM_CHECKPOINT
            )
        
        logger.info(f"Grounded-SAM using device: {self.device}")
        logger.info(f"Grounding DINO checkpoint: {self.grounding_dino_checkpoint}")
        logger.info(f"SAM checkpoint: {self.sam_checkpoint}")
        
        # Load models
        self.grounding_dino_model = self._load_grounding_dino()
        self.sam_predictor = self._load_sam()
    
    def _resolve_path(self, path: Optional[str], default: str) -> str:
        """Resolve model path, checking multiple locations."""
        if path and Path(path).exists():
            return str(path)
        
        # Check project root first (for config files like GroundingDINO config)
        project_root_path = project_root / default
        if project_root_path.exists():
            return str(project_root_path)
        
        # Check weights directory (for model checkpoints)
        for check_dir in [WEIGHTS_DIR, project_root / 'weight', project_root / 'weights']:
            check_path = check_dir / default
            if check_path.exists():
                return str(check_path)
        
        # Return the project root path for config files, weights dir path for checkpoints
        if default.endswith('.py'):  # Config files
            return str(project_root / default)
        return str(WEIGHTS_DIR / default)  # Model weights
    
    def _load_grounding_dino(self):
        """Load Grounding DINO model."""
        try:
            from groundingdino.util.inference import load_model
            
            model = load_model(
                self.grounding_dino_config,
                self.grounding_dino_checkpoint,
                device=self.device,
            )
            logger.info("Grounding DINO model loaded successfully")
            return model
        except ImportError as e:
            raise ImportError(
                "Grounding DINO not installed. Install with:\n"
                "pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO: {e}")
            raise
    
    def _load_sam(self):
        """Load SAM or SAM-HQ predictor."""
        try:
            if self.use_sam_hq:
                # SAM-HQ
                from segment_anything_hq import sam_hq_model_registry, SamPredictor
                sam_type = 'vit_h'
                sam = sam_hq_model_registry[sam_type](checkpoint=self.sam_checkpoint)
            else:
                # Standard SAM
                from segment_anything import sam_model_registry, SamPredictor
                sam_type = 'vit_h'
                sam = sam_model_registry[sam_type](checkpoint=self.sam_checkpoint)
            
            sam.to(device=self.device)
            predictor = SamPredictor(sam)
            logger.info(f"SAM{'HQ' if self.use_sam_hq else ''} model loaded successfully")
            return predictor
        except ImportError as e:
            raise ImportError(
                "SAM not installed. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            raise
    
    def _load_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, Image.Image]:
        """Load image for processing."""
        image_path = str(image_path)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.open(image_path).convert("RGB")
        return image_rgb, image_pil
    
    def _detect_with_grounding_dino(
        self,
        image_pil: Image.Image,
        text_prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Run Grounding DINO detection.
        
        Returns:
            Tuple of (boxes, logits, phrases)
        """
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T
        
        # Transform image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image_pil, None)
        
        # Run prediction
        boxes, logits, phrases = predict(
            model=self.grounding_dino_model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        
        return boxes, logits, phrases
    
    def _segment_with_sam(
        self,
        image: np.ndarray,
        boxes: torch.Tensor,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Run SAM segmentation on detected boxes.
        
        Returns:
            Tuple of (masks, scores)
        """
        self.sam_predictor.set_image(image)
        
        H, W = image.shape[:2]
        
        # Convert normalized boxes to absolute coordinates
        boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
        
        # Convert from cxcywh to xyxy format if needed
        # Grounding DINO outputs in cxcywh format
        boxes_xyxy = self._box_cxcywh_to_xyxy(boxes_xyxy)
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image.shape[:2]
        ).to(self.device)
        
        masks, iou_scores, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        return masks.cpu().numpy(), iou_scores.cpu().numpy().flatten().tolist()
    
    def _box_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from center-x,center-y,width,height to x1,y1,x2,y2."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def detect_and_segment(
        self,
        image_path: Union[str, Path, np.ndarray],
        text_prompt: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Run complete Grounded-SAM pipeline: detect + segment.
        
        This is the main entry point that combines Grounding DINO
        detection with SAM segmentation.
        
        Args:
            image_path: Path to input image or numpy array
            text_prompt: Text description for detection (e.g., "stone contaminant")
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary containing:
                - masks: List of segmentation masks
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - phrases: List of detected class labels
                - scores: List of detection confidence scores
                - iou_scores: List of SAM IoU scores
                - image_size: Tuple of (height, width)
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image_rgb, image_pil = self._load_image(image_path)
            source_path = Path(image_path)
        else:
            image_rgb = image_path
            image_pil = Image.fromarray(image_path)
            source_path = None
        
        H, W = image_rgb.shape[:2]
        
        # Step 1: Detect with Grounding DINO
        logger.info(f"Detecting with prompt: '{text_prompt}'")
        boxes, logits, phrases = self._detect_with_grounding_dino(image_pil, text_prompt)
        
        if len(boxes) == 0:
            logger.warning(f"No objects detected with prompt: '{text_prompt}'")
            return {
                'masks': [],
                'boxes': [],
                'phrases': [],
                'scores': [],
                'iou_scores': [],
                'image_size': (H, W),
                'source': str(source_path) if source_path else None,
            }
        
        # Step 2: Segment with SAM
        logger.info(f"Segmenting {len(boxes)} detected objects")
        masks, iou_scores = self._segment_with_sam(image_rgb, boxes)
        
        # Convert boxes to absolute xyxy format
        boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
        boxes_xyxy = self._box_cxcywh_to_xyxy(boxes_xyxy).numpy().tolist()
        
        # Prepare results
        results = {
            'masks': [mask.squeeze().astype(np.uint8) for mask in masks],
            'boxes': boxes_xyxy,
            'phrases': phrases,
            'scores': logits.cpu().numpy().tolist(),
            'iou_scores': iou_scores,
            'image_size': (H, W),
            'source': str(source_path) if source_path else None,
            'text_prompt': text_prompt,
        }
        
        logger.info(f"Detected {len(results['masks'])} objects")
        
        # Save results if output directory specified
        if output_dir:
            self._save_results(results, image_rgb, output_dir, source_path)
        
        return results
    
    def _save_results(
        self,
        results: Dict,
        image: np.ndarray,
        output_dir: Union[str, Path],
        source_path: Optional[Path],
    ):
        """Save detection and segmentation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = source_path.stem if source_path else "image"
        
        # Save masks
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        for i, mask in enumerate(results['masks']):
            mask_path = masks_dir / f"{base_name}_mask_{i}.png"
            cv2.imwrite(str(mask_path), mask * 255)
        
        # Save annotated image
        annotated = self._create_annotated_image(
            image.copy(),
            results['masks'],
            results['boxes'],
            results['phrases'],
            results['scores'],
        )
        annotated_path = output_dir / f"{base_name}_annotated.png"
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        
        # Save JSON results
        json_results = {
            'boxes': results['boxes'],
            'phrases': results['phrases'],
            'scores': results['scores'],
            'iou_scores': results['iou_scores'],
            'image_size': results['image_size'],
            'text_prompt': results['text_prompt'],
        }
        json_path = output_dir / f"{base_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to: {output_dir}")
    
    def _create_annotated_image(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        boxes: List[List[float]],
        phrases: List[str],
        scores: List[float],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create annotated visualization with masks and boxes."""
        # Generate distinct colors
        np.random.seed(42)
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
                  for _ in range(len(masks))]
        
        # Draw masks
        for mask, color in zip(masks, colors):
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
        # Draw boxes and labels
        for box, phrase, score, color in zip(boxes, phrases, scores, colors):
            x1, y1, x2, y2 = [int(c) for c in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{phrase}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def get_highest_confidence_detection(self, results: Dict) -> Optional[Dict]:
        """
        Get the detection with highest confidence score.
        
        Useful for Phase 1 reference mask generation.
        """
        if not results['scores']:
            return None
        
        max_idx = np.argmax(results['scores'])
        
        return {
            'mask': results['masks'][max_idx],
            'box': results['boxes'][max_idx],
            'phrase': results['phrases'][max_idx],
            'score': results['scores'][max_idx],
            'iou_score': results['iou_scores'][max_idx],
            'image_size': results['image_size'],
        }
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        text_prompt: str,
        output_dir: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Process multiple images.
        
        Args:
            image_paths: List of image paths
            text_prompt: Text description for detection
            output_dir: Optional directory to save results
            show_progress: Show progress bar
            
        Returns:
            List of result dictionaries
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(image_paths, desc="Processing") if show_progress else image_paths
        
        for image_path in iterator:
            try:
                result = self.detect_and_segment(
                    image_path,
                    text_prompt,
                    output_dir=output_dir,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'masks': [],
                    'boxes': [],
                    'error': str(e),
                    'source': str(image_path),
                })
        
        total_detections = sum(len(r.get('masks', [])) for r in results)
        logger.info(f"Processed {len(results)} images, found {total_detections} detections")
        
        return results


def run_grounded_sam(
    input_image: Union[str, Path],
    text_prompt: str,
    output_dir: str = "outputs",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    use_sam_hq: bool = False,
    device: str = "cuda",
) -> Dict:
    """
    Convenience function matching the grounded_sam_demo.py interface.
    
    Equivalent to running:
        python grounded_sam_demo.py \\
          --input_image {input_image} \\
          --text_prompt {text_prompt} \\
          --output_dir {output_dir} \\
          --box_threshold {box_threshold} \\
          --text_threshold {text_threshold}
    """
    gsam = GroundedSAM(
        use_sam_hq=use_sam_hq,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    
    return gsam.detect_and_segment(
        image_path=input_image,
        text_prompt=text_prompt,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Grounded-SAM: Detect and Segment with Text Prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python grounded_sam.py \\
        --input data/raw/sample.jpg \\
        --prompt "stone contaminant" \\
        --output outputs/ \\
        --box-threshold 0.3 \\
        --text-threshold 0.25
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="Text prompt for detection")
    parser.add_argument("--output", "-o", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--box-threshold", type=float, default=0.3,
                        help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Text matching threshold")
    parser.add_argument("--use-sam-hq", action="store_true",
                        help="Use SAM-HQ instead of standard SAM")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Computation device")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Grounded-SAM Detection & Segmentation")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Box threshold: {args.box_threshold}")
    print(f"Text threshold: {args.text_threshold}")
    print(f"Using SAM-HQ: {args.use_sam_hq}")
    print("=" * 60)
    
    results = run_grounded_sam(
        input_image=args.input,
        text_prompt=args.prompt,
        output_dir=args.output,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        use_sam_hq=args.use_sam_hq,
        device=args.device,
    )
    
    print(f"\nDetected {len(results['masks'])} objects")
    for i, (phrase, score) in enumerate(zip(results['phrases'], results['scores'])):
        print(f"  {i+1}. {phrase}: confidence={score:.3f}")
    print(f"\nResults saved to: {args.output}")
