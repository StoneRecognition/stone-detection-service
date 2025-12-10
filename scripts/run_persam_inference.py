
import os
import sys
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Setup Paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
PERSAM_DIR = ROOT_DIR / "Personalize-SAM"
sys.path.append(str(PERSAM_DIR))

try:
    from per_segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Error: Could not import per_segment_anything.")
    sys.exit(1)

# Mask Weights Class
class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def point_selection(mask_sim, topk=1):
    # mask_sim: (H, W) tensor
    # Find local maxima to avoid clustering points on the same object
    w, h = mask_sim.shape
    
    # 1. Max Pool to find peaks (Window size 15x15)
    # This suppression radius ensures we pick points at least ~7-8 pixels apart
    x = mask_sim.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    max_out = F.max_pool2d(x, kernel_size=15, stride=1, padding=7)
    keep = (x == max_out).float()
    
    # 2. Filter map
    x_sparse = x * keep
    x_flat = x_sparse.flatten()
    
    # 3. TopK on distinct peaks
    # We might have fewer non-zero peaks than topk, but topk retrieves largest anyway
    vals, indices = x_flat.topk(min(topk, x_flat.shape[0]))
    
    topk_y = indices // w
    topk_x = indices % w
    
    topk_xy = torch.stack((topk_x, topk_y), dim=1) # (K, 2)
    topk_label = np.array([1] * len(indices))
    topk_xy = topk_xy.cpu().numpy()
    
    return topk_xy, topk_label

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0.0
    return intersection / union

def merge_nms(masks, scores, iou_thresh=0.5, merge_thresh=0.8):
    """
    Custom NMS:
    - If IoU > merge_thresh: Merge masks (Union)
    - If IoU > iou_thresh: Suppress lower score
    - Else: Keep both
    """
    # Sort by score descending
    indices = np.argsort(scores)[::-1]
    kept_masks = []
    
    for idx in indices:
        current_mask = masks[idx]
        is_merged = False
        is_suppressed = False
        
        for k_idx, k_mask in enumerate(kept_masks):
            iou = calculate_iou(current_mask, k_mask)
            
            if iou > merge_thresh:
                # Merge into the existing kept mask
                kept_masks[k_idx] = np.logical_or(k_mask, current_mask)
                is_merged = True
                break
            elif iou > iou_thresh:
                # Standard NMS suppression
                is_suppressed = True
                break
        
        if not is_merged and not is_suppressed:
            kept_masks.append(current_mask)
            
    return kept_masks

def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # 1. Load Custom Weights
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return

    print(f"Loading custom weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device)
    
    target_feat = checkpoint['target_feat'].to(device)
    mask_weights_state = checkpoint['mask_weights']
    
    # 2. Load SAM
    sam_type = checkpoint.get('sam_type', 'vit_t')
    print(f"Loading {sam_type}...")
    sam = sam_model_registry[sam_type](checkpoint=args.mobile_sam).to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    # 3. Weights
    mask_weights = Mask_Weights().to(device)
    mask_weights.load_state_dict(mask_weights_state)
    mask_weights.eval()
    
    # 4. Processing Loop
    input_path = Path(args.image)
    if input_path.is_dir():
        image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    else:
        image_files = [input_path]

    for img_path in image_files:
        print(f"[{img_path.name}] Processing...")
        image = cv2.imread(str(img_path))
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor.set_image(image_rgb)
        test_feat = predictor.features.squeeze()

        # Similarity Map
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
        
        sim = target_feat @ test_feat
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()

        # Selection
        topk_xy, topk_label = point_selection(sim, topk=args.k)
        
        # Prepare weighting
        mw = mask_weights.weights.detach().cpu().numpy()
        w_sum = np.sum(mw)
        final_weights = np.array([1-w_sum, mw[0,0], mw[1,0]])

        candidate_masks = []
        candidate_scores = []
        
        # Batch prediction? SamPredictor supports batch points but one image.
        # But predict() takes point_coords (N, 2), assumes they are for ONE mask?
        # No, predict(point_coords, multimask=True) returns (3, H, W) for inputs.
        # If we pass multiple points, SAM tries to find ONE object defined by those points usually.
        # PerSAM iterates points one by one. We stick to that for correctness.

        for i in range(args.k):
            coords = topk_xy[i:i+1]
            labels = topk_label[i:i+1]
            
            masks, scores, logits, logits_high = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=True)
            
            if isinstance(logits_high, torch.Tensor):
                logits_high = logits_high.detach().cpu().numpy()

            weighted_logits = logits_high * final_weights[:, None, None]
            final_logit = weighted_logits.sum(0)
            final_mask = final_logit > 0
            
            if final_mask.sum() < 20: continue # Skip noise
            
            # Score
            score = np.max(scores)
            
            candidate_masks.append(final_mask)
            candidate_scores.append(score)

        print(f" - Found {len(candidate_masks)} candidates. Merging...")
        
        # Apply Custom NMS
        final_instances = merge_nms(candidate_masks, candidate_scores, iou_thresh=args.nms, merge_thresh=args.merge_thresh)
        print(f" - Final Instances: {len(final_instances)}")
        
        # Visualization
        plt.figure(figsize=(12, 12))
        plt.imshow(image_rgb)
        
        # Plot contours for 'boundaries'
        colors = plt.cm.rainbow(np.linspace(0, 1, len(final_instances)))
        
        for idx, mask in enumerate(final_instances):
            # Mask overlay
            color = colors[idx]
            # show_mask(mask, plt.gca(), color=color)
            
            # Contour
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                poly = cnt.reshape(-1, 2)
                plt.plot(poly[:,0], poly[:,1], color=color, linewidth=2)
                
            # Centroid for label
            M = cv2.moments(mask_uint8)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                plt.text(cX, cY, str(idx+1), color='white', fontsize=10, fontweight='bold', bbox=dict(facecolor=color, alpha=0.5))

        out_path = Path("predictions") / f"persam_{img_path.name}"
        out_path.parent.mkdir(exist_ok=True)
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {out_path}")

def show_mask(mask, ax, color):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image[..., 3] = 0.4 # Alpha
    ax.imshow(mask_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default="weight/persam_f_stone.pt")
    parser.add_argument("--mobile_sam", type=str, default="weight/mobile_sam.pt")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--nms", type=float, default=0.5, help="Intersection over Union Threshold")
    parser.add_argument("--merge_thresh", type=float, default=0.34, help="IoU to Merge")
    args = parser.parse_args()
    run_inference(args)
