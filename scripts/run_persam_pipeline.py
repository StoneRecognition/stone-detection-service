
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from PIL import Image
import torchvision.ops as ops
import matplotlib.pyplot as plt

# Setup Paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "Personalize-SAM"))
sys.path.append(str(ROOT_DIR / "GroundingDINO"))
sys.path.append(str(ROOT_DIR / "segment-anything")) # Standard SAM

try:
    from per_segment_anything import sam_model_registry, SamPredictor
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry as std_sam_registry
except ImportError:
    print("Error: Could not import SAM modules.")
    sys.exit(1)

# Mask Weights Class (Must match training)
class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', lw=2))

def run_grounding_dino(image_path, text_prompt="stone. rock."):
    """Run GroundingDINO to get candidate boxes"""
    from groundingdino.util.inference import load_model, predict, annotate
    import groundingdino.datasets.transforms as T
    
    config_path = ROOT_DIR / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = ROOT_DIR / "weight/groundingdino_swint_ogc.pth"
    
    model = load_model(str(config_path), str(weights_path))
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image_src = Image.open(image_path).convert("RGB")
    image_transformed, _ = transform(image_src, None)
    
    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=0.25,
        text_threshold=0.25
    )
    
    # Convert boxes to xyxy absolute
    w, h = image_src.size
    boxes_abs = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = ops.box_convert(boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")
    
    return boxes_xyxy.numpy()

def load_persam_model(ckpt_path, sam_type, device):
    print(f"Loading PerSAM ({sam_type}) from {ckpt_path}...")
    
    # Load PerSAM weights
    if not os.path.exists(ckpt_path):
        print(f"Target weights {ckpt_path} NOT FOUND.")
        return None, None, None, None

    checkpoint = torch.load(ckpt_path, map_location=device)
    target_feat = checkpoint['target_feat'].to(device)
    mask_weights_state = checkpoint['mask_weights']
    
    # Load Backbone
    # Using Standard SAM registry or PerSAM one?
    # PerSAM wraps SAM.
    sam = sam_model_registry[sam_type](checkpoint=str(ROOT_DIR / "weight/sam_vit_h_4b8939.pth"))
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    
    mask_weights = Mask_Weights().to(device)
    mask_weights.load_state_dict(mask_weights_state)
    mask_weights.eval()
    
    return predictor, mask_weights, target_feat, sam

def run_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    
    # 1. Load PerSAM
    predictor, mask_weights, target_feat, sam_model = load_persam_model(args.weights, args.sam_type, device)
    if predictor is None: return

    # 2. Iterate Images
    input_path = Path(args.image_dir)
    images = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    
    output_dir = Path("outputs/pipeline_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in images:
        print(f"\nProcessing {img_file.name}...")
        image = cv2.imread(str(img_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        candidates_masks = []
        candidates_scores = []
        
        # --- Strategy A: GroundingDINO Boxes ---
        print("Running GroundingDINO...")
        dino_boxes = run_grounding_dino(str(img_file))
        print(f" - Found {len(dino_boxes)} box proposals")
        
        predictor.set_image(image_rgb)
        
        # Refine DINO boxes with PerSAM
        for box in dino_boxes:
            # Use box as prompt
            masks, scores, logits = predictor.predict(
                box=box[None, :],
                multimask_output=True
            )
            # PerSAM filtering logic could define best mask, but here let's take best score
            best_idx = np.argmax(scores)
            candidates_masks.append(masks[best_idx])
            candidates_scores.append(scores[best_idx])

        # --- Strategy B: Segment Everything (SAM Grid) ---
        print("Running Segment Everything...")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )
        
        grid_masks = mask_generator.generate(image_rgb)
        print(f" - Found {len(grid_masks)} grid masks")
        
        # Filter Grid Masks using PerSAM Feature matching?
        # Or just keep them? User said "run both modes".
        # We can score them against PerSAM target_feat!
        
        # Feature Match Scoring
        # Need encoding? mask_generator doesn't return features easily unless modified.
        # But we can assume high quality masks are good.
        
        for m in grid_masks:
            m_bool = m['segmentation'] 
            candidates_masks.append(m_bool)
            candidates_scores.append(m['predicted_iou']) # Low confidence compare to SAM?
            
        
        # --- Merge & Filter ---
        print("Merging Detection Streams...")
        # Convert all to common format
        # Apply NMS
        
        final_kept = []
        # Sort by score
        indices = np.argsort(candidates_scores)[::-1]
        
        used_mask = np.zeros(image.shape[:2], dtype=bool)
        
        for idx in indices:
            mask = candidates_masks[idx]
            
            # IoU with already selected
            overlap = np.logical_and(mask, used_mask).sum()
            area = mask.sum()
            if area == 0: continue
            
            if overlap / area > 0.4: 
                continue # Duplicate
            
            final_kept.append(mask)
            used_mask = np.logical_or(used_mask, mask)
            
        print(f"Final Objects: {len(final_kept)}")
        
        # Visualization
        plt.figure(figsize=(12, 12))
        plt.imshow(image_rgb)
        
        # Draw Contours & Boxes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(final_kept)))
        for i, mask in enumerate(final_kept):
            color = colors[i]
            
            # Contour
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                poly = cnt.reshape(-1, 2)
                plt.plot(poly[:,0], poly[:,1], color=color, linewidth=2)
        
        plt.axis('off')
        out_file = output_dir / f"pipeline_{img_file.name}"
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/raw")
    parser.add_argument("--weights", type=str, default="weight/persam_f_stone_h.pt") # Using H weights
    parser.add_argument("--sam_type", type=str, default="vit_h")
    args = parser.parse_args()
    run_pipeline(args)
