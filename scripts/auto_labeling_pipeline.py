#!/usr/bin/env python3
"""
Multi-Model Ensemble Auto-Labeling Pipeline

Combines GroundingDINO, SAM variants, RAM, and other models for
maximum stone detection accuracy with COCO/YOLO format output.

Usage:
    python scripts/auto_labeling_pipeline.py --input data/raw --output datasets/stones
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torchvision

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "GroundingDINO"))
sys.path.insert(0, str(Path(__file__).parent.parent / "recognize-anything"))
sys.path.insert(0, str(Path(__file__).parent.parent / "sam-hq"))

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the auto-labeling pipeline."""
    
    # Detection prompts focused on INDIVIDUAL stone objects (not gravel background)
    prompts: List[str] = field(default_factory=lambda: [
        "stone",           # Primary target
        "rock",            # Alternative term
        "pebble",          # Small stones
        "stone contaminant",  # Foreign stone
        "foreign object",  # Contaminant
        "debris",          # Debris pieces
    ])
    
    # Detection thresholds (low for max recall)
    box_threshold: float = 0.15
    text_threshold: float = 0.15
    
    # NMS settings
    nms_threshold: float = 0.5
    
    # Mask settings
    min_mask_area: int = 50
    min_contour_points: int = 6
    
    # Ensemble voting threshold (min number of models agreeing)
    voting_threshold: int = 1  # Start with 1 for max recall
    
    # Models to use
    use_sam_hq: bool = True
    use_ram: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: str = "weight/groundingdino_swint_ogc.pth"
    sam_checkpoint: str = "weight/sam_vit_h_4b8939.pth"
    ram_checkpoint: str = "weight/ram_swin_large_14m.pth"


# ============================================================================
# COCO Utilities
# ============================================================================

def create_coco_structure() -> Dict[str, Any]:
    """Create empty COCO dataset structure."""
    return {
        "info": {
            "description": "Stone Detection Dataset - Auto-labeled",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Multi-Model Ensemble Pipeline",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "stone", "supercategory": "object"}
        ]
    }


def mask_to_polygon(mask: np.ndarray, min_points: int = 6) -> Optional[List[float]]:
    """Convert binary mask to polygon coordinates."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    if len(largest) < min_points:
        return None
    
    # Flatten to [x1, y1, x2, y2, ...]
    polygon = largest.flatten().tolist()
    return polygon


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    """Get bounding box [x, y, width, height] from mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


# ============================================================================
# Ensemble Detector
# ============================================================================

class EnsembleDetector:
    """
    Multi-model ensemble detector combining:
    - GroundingDINO for text-guided detection
    - SAM / SAM-HQ for segmentation
    - RAM for automatic tagging
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        
        print(f"Initializing EnsembleDetector on {self.device}...")
        
        # Load models
        self.grounding_dino = self._load_grounding_dino()
        self.sam_predictor = self._load_sam()
        if config.use_ram:
            self.ram_model = self._load_ram()
        else:
            self.ram_model = None
            
        print("All models loaded successfully!")
    
    def _load_grounding_dino(self):
        """Load GroundingDINO model."""
        from groundingdino.util.inference import load_model
        
        print("Loading GroundingDINO...")
        model = load_model(
            self.config.grounding_dino_config,
            self.config.grounding_dino_checkpoint,
            device=self.device
        )
        # Force float32 to avoid type mismatch
        model = model.float().to(self.device)
        return model
    
    def _load_sam(self):
        """Load SAM model."""
        from segment_anything import sam_model_registry, SamPredictor
        
        print("Loading SAM...")
        sam = sam_model_registry["vit_h"](checkpoint=self.config.sam_checkpoint)
        sam.to(device=self.device)
        return SamPredictor(sam)
    
    def _load_ram(self):
        """Load RAM model for automatic tagging."""
        try:
            from ram.models import ram
            from ram import inference_ram
            
            print("Loading RAM...")
            model = ram(pretrained=self.config.ram_checkpoint, 
                       image_size=384, 
                       vit='swin_l')
            model.eval()
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Warning: Could not load RAM model: {e}")
            return None
    
    def get_auto_tags(self, image_pil) -> List[str]:
        """Get automatic tags from RAM model."""
        if self.ram_model is None:
            return []
        
        try:
            from ram import inference_ram
            import torchvision.transforms as transforms
            
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize
            ])
            
            image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                tags, _ = inference_ram(image_tensor, self.ram_model)
            
            return tags.split(" | ") if tags else []
        except Exception as e:
            print(f"Warning: RAM inference failed: {e}")
            return []
    
    def detect_with_prompt(
        self, 
        image: np.ndarray, 
        prompt: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run GroundingDINO detection with a single prompt."""
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T
        from PIL import Image
        
        # Transform image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pil, None)
        
        # Run detection
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=image_transformed,
                caption=prompt,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                device=self.device
            )
        
        return boxes.cpu().numpy(), logits.cpu().numpy(), phrases
    
    def segment_boxes(
        self, 
        image: np.ndarray, 
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """Segment detected boxes using SAM."""
        if len(boxes) == 0:
            return []
        
        # Set image for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        
        h, w = image.shape[:2]
        masks = []
        
        for box in boxes:
            # Convert from center format to xyxy
            cx, cy, bw, bh = box
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            box_xyxy = np.array([x1, y1, x2, y2])
            
            # Get SAM prediction
            mask_predictions, scores, _ = self.sam_predictor.predict(
                box=box_xyxy,
                multimask_output=True
            )
            
            # Take best mask
            best_idx = np.argmax(scores)
            masks.append(mask_predictions[best_idx])
        
        return masks
    
    def detect_all_stones(self, image_path: str) -> Dict[str, Any]:
        """
        Run full ensemble detection pipeline.
        
        Returns:
            Dictionary with masks, boxes, scores, and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Collect all detections from multiple prompts
        all_boxes = []
        all_scores = []
        all_phrases = []
        
        # Run detection with each prompt
        prompts_to_use = self.config.prompts.copy()
        
        # Add RAM-discovered tags if available
        if self.ram_model is not None:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            auto_tags = self.get_auto_tags(image_pil)
            stone_related = [t for t in auto_tags if any(
                kw in t.lower() for kw in ['stone', 'rock', 'mineral', 'gravel', 'pebble', 'rubble']
            )]
            prompts_to_use.extend(stone_related)
        
        print(f"  Using prompts: {prompts_to_use}")
        
        for prompt in prompts_to_use:
            try:
                boxes, scores, phrases = self.detect_with_prompt(image, prompt + ".")
                if len(boxes) > 0:
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_phrases.extend(phrases)
            except Exception as e:
                print(f"  Warning: Detection failed for prompt '{prompt}': {e}")
        
        if not all_boxes:
            return {
                "masks": [],
                "boxes": np.array([]),
                "scores": np.array([]),
                "phrases": [],
                "image_size": (h, w)
            }
        
        # Combine all detections
        combined_boxes = np.vstack(all_boxes)
        combined_scores = np.concatenate(all_scores)
        
        # Apply NMS to remove duplicates
        boxes_xyxy = np.zeros((len(combined_boxes), 4))
        for i, box in enumerate(combined_boxes):
            cx, cy, bw, bh = box
            boxes_xyxy[i] = [
                (cx - bw/2) * w,
                (cy - bh/2) * h,
                (cx + bw/2) * w,
                (cy + bh/2) * h
            ]
        
        keep_indices = torchvision.ops.nms(
            torch.tensor(boxes_xyxy, dtype=torch.float32),
            torch.tensor(combined_scores, dtype=torch.float32),
            self.config.nms_threshold
        ).numpy()
        
        final_boxes = combined_boxes[keep_indices]
        final_scores = combined_scores[keep_indices]
        
        print(f"  Detected {len(combined_boxes)} boxes, {len(final_boxes)} after NMS")
        
        # Segment each box with SAM
        masks = self.segment_boxes(image, final_boxes)
        
        return {
            "masks": masks,
            "boxes": final_boxes,
            "scores": final_scores,
            "phrases": all_phrases,
            "image_size": (h, w)
        }


# ============================================================================
# Pipeline Runner
# ============================================================================

class AutoLabelingPipeline:
    """Main pipeline for batch processing images."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = EnsembleDetector(config)
        
    def process_image(
        self, 
        image_path: str, 
        image_id: int
    ) -> Tuple[Dict, List[Dict]]:
        """Process a single image and return COCO entries."""
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Create image entry
        image_entry = {
            "id": image_id,
            "file_name": Path(image_path).name,
            "width": w,
            "height": h,
        }
        
        # Run detection
        results = self.detector.detect_all_stones(image_path)
        
        # Create annotations
        annotations = []
        for i, mask in enumerate(results["masks"]):
            # Get polygon
            polygon = mask_to_polygon(mask, self.config.min_contour_points)
            if polygon is None:
                continue
            
            # Get bbox
            bbox = mask_to_bbox(mask)
            area = float(mask.sum())
            
            if area < self.config.min_mask_area:
                continue
            
            ann = {
                "id": None,  # Set later
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [polygon],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
            
            # Add confidence if available
            if i < len(results["scores"]):
                ann["score"] = float(results["scores"][i])
            
            annotations.append(ann)
        
        return image_entry, annotations
    
    def process_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """Process all images in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"  # For YOLO format
        vis_dir = output_path / "visualizations"
        
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        if visualize:
            vis_dir.mkdir(exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} images to process")
        
        # Create COCO structure
        coco_data = create_coco_structure()
        annotation_id = 1
        
        # Process each image
        for idx, image_file in enumerate(image_files):
            print(f"\nProcessing [{idx+1}/{len(image_files)}]: {image_file.name}")
            
            try:
                image_entry, annotations = self.process_image(
                    str(image_file), 
                    image_id=idx + 1
                )
                
                # Add to COCO data
                coco_data["images"].append(image_entry)
                
                for ann in annotations:
                    ann["id"] = annotation_id
                    coco_data["annotations"].append(ann)
                    annotation_id += 1
                
                # Copy image to output
                import shutil
                shutil.copy(image_file, images_dir / image_file.name)
                
                # Create visualization
                if visualize and annotations:
                    self._create_visualization(
                        str(image_file),
                        annotations,
                        str(vis_dir / f"vis_{image_file.name}")
                    )
                
                # Create YOLO label file
                self._save_yolo_labels(
                    annotations,
                    image_entry["width"],
                    image_entry["height"],
                    labels_dir / f"{image_file.stem}.txt"
                )
                
                print(f"  Found {len(annotations)} stone annotations")
                
            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}")
        
        # Save COCO JSON
        coco_path = output_path / "annotations.json"
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        print(f"  COCO JSON: {coco_path}")
        print(f"  YOLO labels: {labels_dir}")
        print(f"{'='*60}")
        
        return coco_data
    
    def _create_visualization(
        self, 
        image_path: str, 
        annotations: List[Dict],
        output_path: str
    ):
        """Create visualization with mask overlays."""
        image = cv2.imread(image_path)
        overlay = image.copy()
        
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, ann in enumerate(annotations):
            color = colors[i % len(colors)]
            
            # Draw segmentation polygon
            if "segmentation" in ann and ann["segmentation"]:
                pts = np.array(ann["segmentation"][0]).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(image, [pts], True, color, 2)
            
            # Draw bbox
            if "bbox" in ann:
                x, y, w, h = [int(v) for v in ann["bbox"]]
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                label = f"stone"
                if "score" in ann:
                    label += f" {ann['score']:.2f}"
                cv2.putText(image, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Blend overlay
        result = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
        cv2.imwrite(output_path, result)
    
    def _save_yolo_labels(
        self, 
        annotations: List[Dict],
        img_width: int,
        img_height: int,
        output_path: Path
    ):
        """Save annotations in YOLO format."""
        lines = []
        for ann in annotations:
            if "bbox" not in ann:
                continue
            
            x, y, w, h = ann["bbox"]
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            # Class 0 for stone
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Ensemble Auto-Labeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all images in data/raw
    python scripts/auto_labeling_pipeline.py --input data/raw --output datasets/stones
    
    # With custom thresholds for higher precision
    python scripts/auto_labeling_pipeline.py --input data/raw --output datasets/stones --box-threshold 0.25
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input directory containing images")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory for dataset")
    parser.add_argument("--box-threshold", type=float, default=0.15,
                       help="Box confidence threshold (default: 0.15)")
    parser.add_argument("--text-threshold", type=float, default=0.15,
                       help="Text matching threshold (default: 0.15)")
    parser.add_argument("--nms-threshold", type=float, default=0.5,
                       help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--no-ram", action="store_true",
                       help="Skip RAM auto-tagging")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
        use_ram=not args.no_ram,
        device=args.device,
    )
    
    # Run pipeline
    pipeline = AutoLabelingPipeline(config)
    pipeline.process_directory(
        args.input,
        args.output,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    # Import PIL here for RAM
    from PIL import Image
    main()
