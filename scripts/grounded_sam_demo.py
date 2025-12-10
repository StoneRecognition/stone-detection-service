#!/usr/bin/env python3
"""
Grounded-SAM Demo Script

Unified command-line interface for all Grounded-SAM variants:
- Basic: Detect + Segment
- Inpaint: Detect + Segment + Inpaint/Replace
- Auto: Automatic tagging with RAM/Tag2Text + Detect + Segment

Usage:
    # Basic detection + segmentation
    python scripts/grounded_sam_demo.py detect \\
        --input data/raw/sample.jpg \\
        --prompt "stone contaminant" \\
        --output outputs/

    # Detection + Segmentation + Inpainting
    python scripts/grounded_sam_demo.py inpaint \\
        --input data/raw/sample.jpg \\
        --detect-prompt "stone contaminant" \\
        --inpaint-prompt "clean conveyor surface" \\
        --output outputs/

    # Automatic labeling with RAM
    python scripts/grounded_sam_demo.py auto \\
        --input data/raw/sample.jpg \\
        --model ram \\
        --output outputs/

References:
    https://github.com/IDEA-Research/Grounded-Segment-Anything
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def cmd_detect(args):
    """Run basic Grounded-SAM detection + segmentation."""
    from src.inference.grounded_sam import GroundedSAM
    
    print("=" * 60)
    print("Grounded-SAM: Detect + Segment")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Box threshold: {args.box_threshold}")
    print(f"Text threshold: {args.text_threshold}")
    print("=" * 60)
    
    gsam = GroundedSAM(
        use_sam_hq=args.use_sam_hq,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    
    results = gsam.detect_and_segment(
        image_path=args.input,
        text_prompt=args.prompt,
        output_dir=args.output,
    )
    
    print(f"\n✓ Detected {len(results['masks'])} objects")
    for i, (phrase, score) in enumerate(zip(results['phrases'], results['scores'])):
        print(f"  {i+1}. {phrase}: confidence={score:.3f}")
    print(f"\nResults saved to: {args.output}")


def cmd_inpaint(args):
    """Run Grounded-SAM with inpainting."""
    from src.inference.grounded_sam_inpaint import GroundedSAMInpaint
    
    print("=" * 60)
    print("Grounded-SAM: Detect + Segment + Inpaint")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Detect: '{args.detect_prompt}'")
    print(f"Inpaint: '{args.inpaint_prompt}'")
    print("=" * 60)
    
    inpainter = GroundedSAMInpaint(
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    
    results = inpainter.detect_segment_inpaint(
        image_path=args.input,
        detect_prompt=args.detect_prompt,
        inpaint_prompt=args.inpaint_prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        output_dir=args.output,
    )
    
    print(f"\n✓ Detected {len(results['detection_masks'])} objects")
    print(f"✓ Inpainting complete")
    print(f"Results saved to: {args.output}")


def cmd_auto(args):
    """Run Grounded-SAM with automatic labeling."""
    from src.inference.grounded_sam_auto import GroundedSAMAuto
    
    print("=" * 60)
    print(f"Grounded-SAM: Automatic Labeling ({args.model.upper()})")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Filter: {args.filter}")
    print("=" * 60)
    
    auto_labeler = GroundedSAMAuto(
        model_type=args.model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    
    results = auto_labeler.auto_label(
        image_path=args.input,
        filter_tags=args.filter,
        output_dir=args.output,
    )
    
    print(f"\n✓ Generated tags: {results['tags_string']}")
    print(f"✓ Filtered to: {results['filtered_tags']}")
    print(f"✓ Detected {len(results['masks'])} objects")
    print(f"Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Grounded-SAM Demo - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Mode to run')
    
    # =========================================================================
    # Detect command (basic Grounded-SAM)
    # =========================================================================
    detect_parser = subparsers.add_parser(
        'detect', 
        help='Detect and segment with text prompt'
    )
    detect_parser.add_argument("--input", "-i", type=str, required=True,
                               help="Input image path")
    detect_parser.add_argument("--prompt", "-p", type=str, required=True,
                               help="Text prompt for detection")
    detect_parser.add_argument("--output", "-o", type=str, default="outputs",
                               help="Output directory")
    detect_parser.add_argument("--box-threshold", type=float, default=0.3,
                               help="Box confidence threshold")
    detect_parser.add_argument("--text-threshold", type=float, default=0.25,
                               help="Text matching threshold")
    detect_parser.add_argument("--use-sam-hq", action="store_true",
                               help="Use SAM-HQ for better quality")
    detect_parser.add_argument("--device", type=str, default="cuda",
                               help="Device (cuda/cpu)")
    
    # =========================================================================
    # Inpaint command
    # =========================================================================
    inpaint_parser = subparsers.add_parser(
        'inpaint',
        help='Detect, segment, and inpaint/replace'
    )
    inpaint_parser.add_argument("--input", "-i", type=str, required=True,
                                help="Input image path")
    inpaint_parser.add_argument("--detect-prompt", type=str, required=True,
                                help="Text prompt for detection")
    inpaint_parser.add_argument("--inpaint-prompt", type=str, required=True,
                                help="Text prompt for inpainting")
    inpaint_parser.add_argument("--negative-prompt", type=str, default="",
                                help="Negative prompt for inpainting")
    inpaint_parser.add_argument("--output", "-o", type=str, default="outputs",
                                help="Output directory")
    inpaint_parser.add_argument("--box-threshold", type=float, default=0.3,
                                help="Box confidence threshold")
    inpaint_parser.add_argument("--text-threshold", type=float, default=0.25,
                                help="Text matching threshold")
    inpaint_parser.add_argument("--steps", type=int, default=50,
                                help="Number of diffusion steps")
    inpaint_parser.add_argument("--device", type=str, default="cuda",
                                help="Device (cuda/cpu)")
    
    # =========================================================================
    # Auto command (RAM/Tag2Text)
    # =========================================================================
    auto_parser = subparsers.add_parser(
        'auto',
        help='Automatic labeling with RAM/Tag2Text'
    )
    auto_parser.add_argument("--input", "-i", type=str, required=True,
                             help="Input image path")
    auto_parser.add_argument("--output", "-o", type=str, default="outputs",
                             help="Output directory")
    auto_parser.add_argument("--model", type=str, default="ram",
                             choices=["ram", "tag2text"],
                             help="Tag generation model")
    auto_parser.add_argument("--filter", nargs="+", default=None,
                             help="Filter to specific tags")
    auto_parser.add_argument("--box-threshold", type=float, default=0.25,
                             help="Box confidence threshold")
    auto_parser.add_argument("--text-threshold", type=float, default=0.2,
                             help="Text matching threshold")
    auto_parser.add_argument("--device", type=str, default="cuda",
                             help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.command == 'detect':
        cmd_detect(args)
    elif args.command == 'inpaint':
        cmd_inpaint(args)
    elif args.command == 'auto':
        cmd_auto(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
