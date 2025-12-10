#!/usr/bin/env python3
"""
Grounded-SAM with RAM/Tag2Text for Automatic Labeling

Zero-shot automatic image labeling without manual text prompts.
Uses RAM (Recognize Anything Model) or Tag2Text to automatically
generate tags, then uses those tags for Grounded-SAM detection.

Usage:
    from src.inference.grounded_sam_auto import GroundedSAMAuto
    
    auto_labeler = GroundedSAMAuto()
    results = auto_labeler.auto_label(
        image_path="sample.jpg",
        model="ram",  # or "tag2text"
    )

Command-line equivalent:
    python grounded_sam_auto.py \\
      --input sample.jpg \\
      --model ram \\
      --output outputs/

References:
    https://github.com/IDEA-Research/Grounded-Segment-Anything
    https://github.com/xinyu1205/recognize-anything
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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

# Default model paths
DEFAULT_RAM_CHECKPOINT = "ram_swin_large_14m.pth"
DEFAULT_TAG2TEXT_CHECKPOINT = "tag2text_swin_14m.pth"


class GroundedSAMAuto:
    """
    Automatic labeling using RAM/Tag2Text + Grounded-SAM.
    
    Pipeline:
    1. Generate tags automatically with RAM or Tag2Text
    2. Use generated tags as prompts for Grounding DINO
    3. Segment detected objects with SAM
    
    This enables fully automatic annotation without manual prompts.
    
    Attributes:
        tag_model: RAM or Tag2Text model for tag generation
        grounded_sam: GroundedSAM instance for detection/segmentation
        model_type: 'ram' or 'tag2text'
    """
    
    def __init__(
        self,
        model_type: str = "ram",
        ram_checkpoint: Optional[str] = None,
        tag2text_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize automatic labeling pipeline.
        
        Args:
            model_type: 'ram' or 'tag2text'
            ram_checkpoint: Path to RAM weights
            tag2text_checkpoint: Path to Tag2Text weights
            device: Computation device
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            iou_threshold: IoU threshold for NMS
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_type = model_type.lower()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        
        # Resolve checkpoint paths
        self.ram_checkpoint = self._resolve_path(ram_checkpoint, DEFAULT_RAM_CHECKPOINT)
        self.tag2text_checkpoint = self._resolve_path(tag2text_checkpoint, DEFAULT_TAG2TEXT_CHECKPOINT)
        
        # Lazy load models
        self._tag_model = None
        self._grounded_sam = None
        
        logger.info(f"GroundedSAMAuto initialized with model: {self.model_type}")
    
    def _resolve_path(self, path: Optional[str], default: str) -> str:
        """Resolve model path."""
        if path and Path(path).exists():
            return str(path)
        
        for check_dir in [WEIGHTS_DIR, project_root / 'weight', project_root / 'weights']:
            check_path = check_dir / default
            if check_path.exists():
                return str(check_path)
        
        return str(WEIGHTS_DIR / default)
    
    @property
    def tag_model(self):
        """Lazy load RAM or Tag2Text model."""
        if self._tag_model is None:
            if self.model_type == "ram":
                self._tag_model = self._load_ram()
            else:
                self._tag_model = self._load_tag2text()
        return self._tag_model
    
    @property
    def grounded_sam(self):
        """Lazy load Grounded-SAM."""
        if self._grounded_sam is None:
            from src.inference.grounded_sam import GroundedSAM
            self._grounded_sam = GroundedSAM(
                device=self.device,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            logger.info("Grounded-SAM loaded")
        return self._grounded_sam
    
    def _load_ram(self):
        """Load RAM (Recognize Anything Model)."""
        try:
            from ram.models import ram
            from ram import inference_ram as inference
            
            model = ram(pretrained=self.ram_checkpoint, image_size=384, vit='swin_l')
            model.eval()
            model.to(self.device)
            
            logger.info(f"RAM model loaded from: {self.ram_checkpoint}")
            return {'model': model, 'inference': inference}
        except ImportError:
            raise ImportError(
                "RAM not installed. Install with:\n"
                "pip install git+https://github.com/xinyu1205/recognize-anything.git"
            )
    
    def _load_tag2text(self):
        """Load Tag2Text model."""
        try:
            from ram.models import tag2text
            from ram import inference_tag2text as inference
            
            model = tag2text(pretrained=self.tag2text_checkpoint, image_size=384, vit='swin_b')
            model.threshold = 0.68  # Default threshold for Tag2Text
            model.eval()
            model.to(self.device)
            
            logger.info(f"Tag2Text model loaded from: {self.tag2text_checkpoint}")
            return {'model': model, 'inference': inference}
        except ImportError:
            raise ImportError(
                "Tag2Text not installed. Install with:\n"
                "pip install git+https://github.com/xinyu1205/recognize-anything.git"
            )
    
    def generate_tags(
        self,
        image_path: Union[str, Path, np.ndarray],
    ) -> Tuple[str, List[str]]:
        """
        Generate tags for an image using RAM or Tag2Text.
        
        Args:
            image_path: Path to image or numpy array
            
        Returns:
            Tuple of (tags_string, tags_list)
            - tags_string: Comma-separated tags for Grounding DINO
            - tags_list: List of individual tags
        """
        # Load and preprocess image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(image_path)
        
        # Get model and inference function
        model = self.tag_model['model']
        inference_fn = self.tag_model['inference']
        
        # Generate tags
        with torch.no_grad():
            tags, _ = inference_fn(image, model)
        
        # Process tags
        if isinstance(tags, str):
            tags_string = tags
            tags_list = [t.strip() for t in tags.split(',') if t.strip()]
        else:
            tags_list = list(tags)
            tags_string = ', '.join(tags_list)
        
        logger.info(f"Generated {len(tags_list)} tags: {tags_string}")
        return tags_string, tags_list
    
    def auto_label(
        self,
        image_path: Union[str, Path, np.ndarray],
        filter_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Automatically label image without manual prompts.
        
        Args:
            image_path: Path to image or numpy array
            filter_tags: Only use these tags (if provided)
            exclude_tags: Exclude these tags
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary containing:
                - tags_string: Generated tags
                - tags_list: List of tags
                - filtered_tags: Tags after filtering
                - detection_results: Per-tag detection results
                - all_masks: All segmentation masks
                - all_boxes: All bounding boxes
                - annotated_image: Visualization
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            source_path = Path(image_path)
            image = cv2.imread(str(source_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            source_path = None
            image_rgb = image_path
        
        # Step 1: Generate tags
        tags_string, tags_list = self.generate_tags(image_rgb)
        
        # Step 2: Filter tags
        filtered_tags = tags_list.copy()
        
        if filter_tags:
            filtered_tags = [t for t in filtered_tags if t.lower() in [f.lower() for f in filter_tags]]
        
        if exclude_tags:
            exclude_lower = [e.lower() for e in exclude_tags]
            filtered_tags = [t for t in filtered_tags if t.lower() not in exclude_lower]
        
        if not filtered_tags:
            logger.warning("No tags remaining after filtering")
            return {
                'tags_string': tags_string,
                'tags_list': tags_list,
                'filtered_tags': [],
                'detection_results': {},
                'all_masks': [],
                'all_boxes': [],
            }
        
        # Step 3: Run Grounded-SAM for each tag (or combined)
        filtered_prompt = '. '.join(filtered_tags)
        logger.info(f"Running detection with prompt: '{filtered_prompt}'")
        
        detection_results = self.grounded_sam.detect_and_segment(
            image_path=image_rgb,
            text_prompt=filtered_prompt,
        )
        
        results = {
            'tags_string': tags_string,
            'tags_list': tags_list,
            'filtered_tags': filtered_tags,
            'detection_prompt': filtered_prompt,
            'masks': detection_results['masks'],
            'boxes': detection_results['boxes'],
            'phrases': detection_results['phrases'],
            'scores': detection_results['scores'],
            'image_size': detection_results['image_size'],
            'source': str(source_path) if source_path else None,
        }
        
        # Save results
        if output_dir:
            self._save_results(results, image_rgb, output_dir, source_path)
        
        return results
    
    def auto_label_for_stones(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Specialized auto-labeling for stone detection.
        
        Filters tags to keep only stone-related categories.
        """
        stone_related = [
            'stone', 'rock', 'pebble', 'gravel', 'mineral',
            'boulder', 'aggregate', 'cobble', 'debris'
        ]
        
        return self.auto_label(
            image_path=image_path,
            filter_tags=stone_related,
            output_dir=output_dir,
        )
    
    def _save_results(
        self,
        results: Dict,
        image: np.ndarray,
        output_dir: Union[str, Path],
        source_path: Optional[Path],
    ):
        """Save detection results."""
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
        cv2.imwrite(
            str(output_dir / f"{base_name}_annotated.png"),
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        )
        
        # Save JSON results
        json_results = {
            'tags_generated': results['tags_list'],
            'tags_filtered': results['filtered_tags'],
            'detection_prompt': results['detection_prompt'],
            'detections': [
                {
                    'phrase': phrase,
                    'score': score,
                    'box': box,
                }
                for phrase, score, box in zip(
                    results['phrases'],
                    results['scores'],
                    results['boxes']
                )
            ],
            'image_size': results['image_size'],
        }
        
        with open(output_dir / f"{base_name}_auto_labels.json", 'w') as f:
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
        """Create annotated visualization."""
        np.random.seed(42)
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                  for _ in range(len(masks))]
        
        for mask, color in zip(masks, colors):
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
        for box, phrase, score, color in zip(boxes, phrases, scores, colors):
            x1, y1, x2, y2 = [int(c) for c in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{phrase}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        filter_tags: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """Process multiple images with automatic labeling."""
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(image_paths, desc="Auto-labeling") if show_progress else image_paths
        
        for img_path in iterator:
            try:
                result = self.auto_label(
                    image_path=img_path,
                    filter_tags=filter_tags,
                    output_dir=output_dir,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({'error': str(e), 'source': str(img_path)})
        
        return results


def run_grounded_sam_auto(
    input_image: Union[str, Path],
    output_dir: str = "outputs",
    model_type: str = "ram",
    filter_tags: Optional[List[str]] = None,
    device: str = "cuda",
) -> Dict:
    """Convenience function for command-line usage."""
    auto_labeler = GroundedSAMAuto(model_type=model_type, device=device)
    return auto_labeler.auto_label(
        image_path=input_image,
        filter_tags=filter_tags,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Grounded-SAM with RAM/Tag2Text Automatic Labeling"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--output", "-o", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="ram",
                        choices=["ram", "tag2text"],
                        help="Tag generation model")
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Filter to specific tags")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Grounded-SAM Automatic Labeling")
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"Filter: {args.filter}")
    print("=" * 60)
    
    results = run_grounded_sam_auto(
        input_image=args.input,
        output_dir=args.output,
        model_type=args.model,
        filter_tags=args.filter,
        device=args.device,
    )
    
    print(f"\nGenerated tags: {results.get('tags_string', 'N/A')}")
    print(f"Detected {len(results.get('masks', []))} objects")
    print(f"Results saved to: {args.output}")
