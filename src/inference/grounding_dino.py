#!/usr/bin/env python3
"""
Grounding DINO Object Detection Module

Zero-shot object detection using natural language text prompts.
Uses Grounding DINO for open-set detection without prior training.

Usage:
    from src.inference.grounding_dino import GroundingDINODetector
    
    detector = GroundingDINODetector()
    detections = detector.detect("stone contaminant", image)
    best_box = detector.select_highest_confidence(detections)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Lazy imports for Grounding DINO (may not be installed)
_groundingdino_available = None


def _check_groundingdino_available() -> bool:
    """Check if Grounding DINO is installed."""
    global _groundingdino_available
    if _groundingdino_available is None:
        try:
            from groundingdino.util.inference import load_model, predict
            _groundingdino_available = True
        except ImportError:
            _groundingdino_available = False
    return _groundingdino_available


# Try to load config
try:
    from src.utils.settings import config
    weights_dir = Path(config.get('paths.weights_dir', 'weight'))
except ImportError:
    weights_dir = Path('./weight')


class GroundingDINODetector:
    """
    Zero-shot object detection using Grounding DINO.
    
    Detects objects in images based on natural language text prompts,
    without requiring prior training on the target objects.
    
    Attributes:
        model: Loaded Grounding DINO model
        device: Computation device (cuda/cpu)
        box_threshold: Minimum confidence for bounding boxes
        text_threshold: Minimum confidence for text matching
    """
    
    # Default model weights path
    DEFAULT_WEIGHTS = "groundingdino_swint_ogc.pth"
    DEFAULT_CONFIG = "GroundingDINO_SwinT_OGC.py"
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        Initialize Grounding DINO detector.
        
        Args:
            weights_path: Path to model weights. If None, uses default.
            config_path: Path to model config. If None, uses default.
            device: Device to use ('cuda', 'cpu', or None for auto).
            box_threshold: Confidence threshold for bounding boxes.
            text_threshold: Confidence threshold for text matching.
        """
        if not _check_groundingdino_available():
            raise ImportError(
                "Grounding DINO is not installed. "
                "Install with: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
            )
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set paths
        self.weights_path = weights_path or str(weights_dir / self.DEFAULT_WEIGHTS)
        self.config_path = config_path
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the Grounding DINO model."""
        from groundingdino.util.inference import load_model
        
        # Get config path from groundingdino package if not provided
        if self.config_path is None:
            import groundingdino
            package_path = Path(groundingdino.__file__).parent
            self.config_path = str(package_path / "config" / self.DEFAULT_CONFIG)
        
        # Verify weights exist
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"Grounding DINO weights not found at: {self.weights_path}\n"
                f"Download from: https://github.com/IDEA-Research/GroundingDINO/releases"
            )
        
        model = load_model(self.config_path, self.weights_path, device=self.device)
        return model
    
    def detect(
        self,
        text_prompt: str,
        image: Union[np.ndarray, str, Path],
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Detect objects matching the text prompt.
        
        Args:
            text_prompt: Natural language description (e.g., "stone contaminant")
            image: Input image (numpy array, file path, or Path object)
            box_threshold: Optional override for box confidence threshold
            text_threshold: Optional override for text confidence threshold
            
        Returns:
            Dictionary containing:
                - boxes: List of bounding boxes [x1, y1, x2, y2] (normalized 0-1)
                - confidences: List of confidence scores
                - labels: List of matched text labels
                - image_size: Tuple of (height, width)
        """
        from groundingdino.util.inference import predict
        from groundingdino.util.utils import get_phrases_from_posmap
        import groundingdino.datasets.transforms as T
        
        box_threshold = box_threshold or self.box_threshold
        text_threshold = text_threshold or self.text_threshold
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size
        height, width = image.shape[:2]
        
        # Prepare image for model
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Convert to PIL for transform
        from PIL import Image
        image_pil = Image.fromarray(image)
        image_transformed, _ = transform(image_pil, None)
        
        # Run prediction
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        
        return {
            "boxes": boxes.cpu().numpy().tolist() if isinstance(boxes, torch.Tensor) else boxes,
            "confidences": logits.cpu().numpy().tolist() if isinstance(logits, torch.Tensor) else logits,
            "labels": phrases,
            "image_size": (height, width),
        }
    
    def detect_with_absolute_boxes(
        self,
        text_prompt: str,
        image: Union[np.ndarray, str, Path],
        **kwargs,
    ) -> Dict:
        """
        Detect objects and return absolute pixel coordinates for boxes.
        
        Same as detect() but converts normalized boxes to pixel coordinates.
        
        Returns:
            Dictionary with boxes as [x1, y1, x2, y2] in pixel coordinates.
        """
        result = self.detect(text_prompt, image, **kwargs)
        
        height, width = result["image_size"]
        absolute_boxes = []
        
        for box in result["boxes"]:
            x_center, y_center, w, h = box
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)
            absolute_boxes.append([x1, y1, x2, y2])
        
        result["boxes_absolute"] = absolute_boxes
        return result
    
    def select_highest_confidence(self, detections: Dict) -> Optional[Dict]:
        """
        Select the detection with highest confidence.
        
        Args:
            detections: Output from detect() or detect_with_absolute_boxes()
            
        Returns:
            Dictionary with highest confidence detection, or None if no detections.
            Contains: box, confidence, label, box_absolute (if available)
        """
        if not detections["confidences"]:
            return None
        
        max_idx = np.argmax(detections["confidences"])
        
        result = {
            "box": detections["boxes"][max_idx],
            "confidence": detections["confidences"][max_idx],
            "label": detections["labels"][max_idx],
            "image_size": detections["image_size"],
        }
        
        if "boxes_absolute" in detections:
            result["box_absolute"] = detections["boxes_absolute"][max_idx]
        
        return result
    
    def get_center_point(self, box: List[float], image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get center point of a bounding box in pixel coordinates.
        
        Args:
            box: Normalized box [x_center, y_center, width, height]
            image_size: Tuple of (height, width)
            
        Returns:
            Tuple of (x, y) center coordinates in pixels
        """
        height, width = image_size
        x_center, y_center = box[0], box[1]
        return (int(x_center * width), int(y_center * height))


def detect_stones(
    image: Union[np.ndarray, str, Path],
    text_prompt: str = "stone contaminant",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: Optional[str] = None,
) -> Dict:
    """
    Convenience function to detect stones in an image.
    
    Args:
        image: Input image (numpy array or path)
        text_prompt: Text description for detection
        box_threshold: Confidence threshold for boxes
        text_threshold: Confidence threshold for text
        device: Computation device
        
    Returns:
        Detection results dictionary
    """
    detector = GroundingDINODetector(
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    return detector.detect_with_absolute_boxes(text_prompt, image)


def get_best_stone_detection(
    image: Union[np.ndarray, str, Path],
    text_prompt: str = "stone contaminant",
    **kwargs,
) -> Optional[Dict]:
    """
    Get the highest confidence stone detection from an image.
    
    Args:
        image: Input image (numpy array or path)
        text_prompt: Text description for detection
        **kwargs: Additional arguments passed to GroundingDINODetector
        
    Returns:
        Best detection or None if no stones found
    """
    detector = GroundingDINODetector(**kwargs)
    detections = detector.detect_with_absolute_boxes(text_prompt, image)
    return detector.select_highest_confidence(detections)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect objects with text prompts")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="stone contaminant",
                        help="Text prompt for detection")
    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Text confidence threshold")
    
    args = parser.parse_args()
    
    print(f"Detecting '{args.prompt}' in {args.image}...")
    
    result = detect_stones(
        args.image,
        text_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    
    print(f"Found {len(result['boxes'])} detections:")
    for i, (box, conf, label) in enumerate(
        zip(result["boxes_absolute"], result["confidences"], result["labels"])
    ):
        print(f"  {i+1}. {label}: confidence={conf:.3f}, box={box}")
