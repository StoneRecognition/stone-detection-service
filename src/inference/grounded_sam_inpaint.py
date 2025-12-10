#!/usr/bin/env python3
"""
Grounded-SAM with Inpainting Module

Detect, segment, and inpaint/replace objects with AI-generated content.
Combines Grounding DINO + SAM + Stable Diffusion Inpainting.

Usage:
    from src.inference.grounded_sam_inpaint import GroundedSAMInpaint
    
    inpainter = GroundedSAMInpaint()
    result = inpainter.detect_segment_inpaint(
        image_path="sample.jpg",
        detect_prompt="stone contaminant",
        inpaint_prompt="clean conveyor belt surface",
    )

Command-line equivalent:
    python grounded_sam_inpaint.py \\
      --input sample.jpg \\
      --detect-prompt "stone contaminant" \\
      --inpaint-prompt "clean surface" \\
      --output outputs/

References:
    https://github.com/IDEA-Research/Grounded-Segment-Anything
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

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


class GroundedSAMInpaint:
    """
    Grounded-SAM with Stable Diffusion Inpainting.
    
    Pipeline:
    1. Detect objects with Grounding DINO (text prompt)
    2. Segment objects with SAM
    3. Inpaint/replace detected regions with Stable Diffusion
    
    Useful for:
    - Removing detected contaminants and replacing with clean surface
    - Object replacement and editing
    - Data augmentation (inpaint variations)
    
    Attributes:
        grounded_sam: GroundedSAM instance for detection/segmentation
        inpaint_pipeline: Stable Diffusion inpainting pipeline
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        inpaint_model: str = "runwayml/stable-diffusion-inpainting",
    ):
        """
        Initialize Grounded-SAM Inpainting pipeline.
        
        Args:
            device: Computation device ('cuda', 'cpu', or None for auto)
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            inpaint_model: Hugging Face model ID for inpainting
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.inpaint_model = inpaint_model
        
        # Lazy load models
        self._grounded_sam = None
        self._inpaint_pipeline = None
    
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
    
    @property
    def inpaint_pipeline(self):
        """Lazy load Stable Diffusion inpainting pipeline."""
        if self._inpaint_pipeline is None:
            try:
                from diffusers import StableDiffusionInpaintPipeline
                
                self._inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.inpaint_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                self._inpaint_pipeline.to(self.device)
                
                # Enable memory optimization
                if self.device == "cuda":
                    self._inpaint_pipeline.enable_attention_slicing()
                
                logger.info(f"Inpainting pipeline loaded: {self.inpaint_model}")
            except ImportError:
                raise ImportError(
                    "diffusers not installed. Install with:\n"
                    "pip install diffusers[torch]"
                )
        return self._inpaint_pipeline
    
    def detect_segment_inpaint(
        self,
        image_path: Union[str, Path, np.ndarray],
        detect_prompt: str,
        inpaint_prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        dilate_mask: int = 15,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Full pipeline: detect, segment, and inpaint.
        
        Args:
            image_path: Path to input image or numpy array
            detect_prompt: Text prompt for detection (e.g., "stone contaminant")
            inpaint_prompt: Text prompt for inpainting (e.g., "clean surface")
            negative_prompt: Negative prompt for inpainting
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            dilate_mask: Pixels to dilate mask for smoother inpainting
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary containing:
                - original_image: Original image
                - detection_masks: List of detected object masks
                - combined_mask: Combined mask of all detections
                - inpainted_image: Result after inpainting
                - detection_results: Full detection details
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
            source_path = Path(image_path)
        else:
            image = Image.fromarray(image_path)
            source_path = None
        
        original_np = np.array(image)
        
        # Step 1: Detect and segment
        logger.info(f"Detecting with prompt: '{detect_prompt}'")
        detection_results = self.grounded_sam.detect_and_segment(
            image_path=original_np,
            text_prompt=detect_prompt,
        )
        
        if not detection_results['masks']:
            logger.warning("No objects detected, returning original image")
            return {
                'original_image': original_np,
                'detection_masks': [],
                'combined_mask': np.zeros(original_np.shape[:2], dtype=np.uint8),
                'inpainted_image': original_np,
                'detection_results': detection_results,
            }
        
        # Step 2: Combine masks
        combined_mask = self._combine_masks(
            detection_results['masks'],
            original_np.shape[:2],
            dilate_kernel=dilate_mask,
        )
        
        # Step 3: Inpaint
        logger.info(f"Inpainting with prompt: '{inpaint_prompt}'")
        inpainted_image = self._inpaint(
            image=image,
            mask=combined_mask,
            prompt=inpaint_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        results = {
            'original_image': original_np,
            'detection_masks': detection_results['masks'],
            'combined_mask': combined_mask,
            'inpainted_image': np.array(inpainted_image),
            'detection_results': detection_results,
        }
        
        # Save results
        if output_dir:
            self._save_results(results, output_dir, source_path)
        
        return results
    
    def _combine_masks(
        self,
        masks: List[np.ndarray],
        shape: Tuple[int, int],
        dilate_kernel: int = 15,
    ) -> np.ndarray:
        """Combine multiple masks into one and optionally dilate."""
        combined = np.zeros(shape, dtype=np.uint8)
        
        for mask in masks:
            combined = np.logical_or(combined, mask > 0).astype(np.uint8)
        
        # Dilate mask for smoother inpainting edges
        if dilate_kernel > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            combined = cv2.dilate(combined, kernel, iterations=1)
        
        return combined * 255  # Convert to 0-255 range
    
    def _inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """Run Stable Diffusion inpainting."""
        # Prepare mask image
        mask_image = Image.fromarray(mask).convert("L")
        
        # Resize to match model requirements (512x512 or 768x768)
        target_size = (512, 512)
        image_resized = image.resize(target_size)
        mask_resized = mask_image.resize(target_size)
        
        # Run inpainting
        result = self.inpaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        # Resize back to original
        result = result.resize(image.size)
        
        return result
    
    def _save_results(
        self,
        results: Dict,
        output_dir: Union[str, Path],
        source_path: Optional[Path],
    ):
        """Save all results to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = source_path.stem if source_path else "image"
        
        # Save original
        cv2.imwrite(
            str(output_dir / f"{base_name}_original.png"),
            cv2.cvtColor(results['original_image'], cv2.COLOR_RGB2BGR)
        )
        
        # Save mask
        cv2.imwrite(
            str(output_dir / f"{base_name}_mask.png"),
            results['combined_mask']
        )
        
        # Save inpainted
        cv2.imwrite(
            str(output_dir / f"{base_name}_inpainted.png"),
            cv2.cvtColor(results['inpainted_image'], cv2.COLOR_RGB2BGR)
        )
        
        # Save comparison (side by side)
        comparison = np.hstack([results['original_image'], results['inpainted_image']])
        cv2.imwrite(
            str(output_dir / f"{base_name}_comparison.png"),
            cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        )
        
        logger.info(f"Results saved to: {output_dir}")
    
    def remove_and_clean(
        self,
        image_path: Union[str, Path],
        detect_prompt: str = "stone contaminant",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Convenience method: detect stones and replace with clean surface.
        
        This is optimized for the stone detection use case.
        """
        return self.detect_segment_inpaint(
            image_path=image_path,
            detect_prompt=detect_prompt,
            inpaint_prompt="clean smooth conveyor belt surface, industrial, no debris",
            negative_prompt="stone, rock, debris, contamination, dirt",
            output_dir=output_dir,
        )


def run_grounded_sam_inpaint(
    input_image: Union[str, Path],
    detect_prompt: str,
    inpaint_prompt: str,
    output_dir: str = "outputs",
    device: str = "cuda",
    **kwargs,
) -> Dict:
    """Convenience function for command-line style usage."""
    inpainter = GroundedSAMInpaint(device=device)
    return inpainter.detect_segment_inpaint(
        image_path=input_image,
        detect_prompt=detect_prompt,
        inpaint_prompt=inpaint_prompt,
        output_dir=output_dir,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Grounded-SAM with Inpainting"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--detect-prompt", type=str, required=True,
                        help="Text prompt for detection")
    parser.add_argument("--inpaint-prompt", type=str, required=True,
                        help="Text prompt for inpainting")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt for inpainting")
    parser.add_argument("--output", "-o", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Grounded-SAM with Inpainting")
    print("=" * 60)
    print(f"Detect: '{args.detect_prompt}'")
    print(f"Inpaint: '{args.inpaint_prompt}'")
    print("=" * 60)
    
    results = run_grounded_sam_inpaint(
        input_image=args.input,
        detect_prompt=args.detect_prompt,
        inpaint_prompt=args.inpaint_prompt,
        output_dir=args.output,
        device=args.device,
        num_inference_steps=args.steps,
        negative_prompt=args.negative_prompt,
    )
    
    print(f"\nDetected {len(results['detection_masks'])} objects")
    print(f"Results saved to: {args.output}")
