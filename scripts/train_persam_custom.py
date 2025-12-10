
import os
import sys
import argparse
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Setup Paths to include Personalize-SAM logic
ROOT_DIR = Path(__file__).parent.parent.absolute()
PERSAM_DIR = ROOT_DIR / "Personalize-SAM"
sys.path.append(str(PERSAM_DIR))

try:
    from per_segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Error: Could not import per_segment_anything. Make sure Personalize-SAM is in the correct path.")
    sys.exit(1)

# Mask Weights Class (Copied from persam_f_multi_obj.py)
class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def calculate_dice_loss(inputs, targets, num_masks=1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def calculate_sigmoid_focal_loss(inputs, targets, num_masks=1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    inputs = inputs.flatten(1)
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def train_and_save(args):
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # Load Model
    print(f"Loading SAM ({args.sam_type}) from {args.ckpt}...")
    sam = sam_model_registry[args.sam_type](checkpoint=args.ckpt).to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    # Directories
    images_path = Path(args.data) / "Images" / "stone"
    masks_path = Path(args.data) / "Annotations" / "stone"
    
    if not images_path.exists():
        print(f"Error: {images_path} does not exist.")
        return

    # List all candidates
    all_images = sorted(list(images_path.glob("*.jpg")))
    print(f"Found {len(all_images)} candidate reference images.")

    best_loss = float('inf')
    best_ref_idx = -1
    best_weights_state = None
    best_target_feat = None
    best_ref_image_path = None

    # Loop through candidates to find BEST one
    # We limit to first 50 to save time if dataset is huge, or run all if args say so
    # The user has ~323. We can just run them, it takes ~10-20 mins.
    # To be safe, let's just run them all or skip bad ones.
    
    # We can check 'learnability' using ONE epoch first? 
    # Or strict logic: Train 100 epochs for EACH, pick best.
    
    start_time = time.time()
    
    valid_count = 0
    
    # Limit to N random samples if too many? No, let's be thorough or assume sorted.
    candidates = all_images
    
    print("Searching for the best reference image (One-Shot Learning)...")

    for idx, ref_image_file in enumerate(tqdm(candidates)):
        ref_idx = idx
        name = ref_image_file.stem
        mask_file = masks_path / (name + ".png")
        
        if not mask_file.exists():
            continue

        # Load
        ref_image = cv2.imread(str(ref_image_file))
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        
        ref_mask = cv2.imread(str(mask_file))
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

        # Preprocess features (SamPredictor)
        try:
             # See predictor.py: returns transformed_mask logic
             ref_mask_feature = predictor.set_image(ref_image, ref_mask)
        except Exception as e:
             # print(f"Skipping {name}: Feature Extraction Error {e}")
             continue
             
        ref_feat = predictor.features.squeeze().permute(1, 2, 0)
        
        # Interpolate Mask to Feature Size
        # ref_mask is coming from predictor set_image return?
        # In original script: ref_mask = predictor.set_image(...)
        # We need to ensure we have the mask tensor for interpolation
        
        # Re-logic from persam_f_multi_obj.py
        # ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear") 
        # But ref_mask from predictor is likely (1, 1, 1024, 1024) un-interpolated?
        # predictor.set_image returns input_mask (1024x1024).
        
        # We need to interpolate it to 64x64 (feature size)
        target_size = ref_feat.shape[0:2] # (64, 64)
        
        # ref_mask_feature is the return value from set_image
        if ref_mask_feature is None:
             continue
             
        # Interpolate
        curr_mask = F.interpolate(ref_mask_feature, size=target_size, mode="bilinear")
        curr_mask = curr_mask.squeeze()[0] # (64, 64)
        
        # Target Feat
        target_feat = ref_feat[curr_mask > 0]
        if target_feat.shape[0] == 0:
            # Skip empty
            continue
            
        target_feat_mean = target_feat.mean(0)
        target_feat_max = torch.max(target_feat, dim=0)[0]
        target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

        # Prepare GT for Loss
        # gt_mask is 1024x1024 boolean
        gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0
        gt_mask = gt_mask.float().unsqueeze(0).flatten(1).to(device)
        
        # PerSAM-F Training (Fine-tuning weights)
        mask_weights = Mask_Weights().to(device)
        mask_weights.train()
        
        optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch_inside)

        # Train Loop (Standard 100 epochs)
        # We optimize to fit THIS reference image perfectly
        
        final_loss = 0.0
        
        # Using SIM (Cosine Similarity) as input to decoder
        # Only need to calculate SIM once per image if ref_feat == target_feat (Self-reconstruction)
        # But wait, PerSAM calculates sim between Test Image and Ref Feat.
        # Here Test Image IS Ref Image.
        
        # Cosine similarity logic
        # ref_feat shape (64, 64, 256)
        # target_feat shape (1, 256)
        
        C = ref_feat.shape[-1]
        h, w = ref_feat.shape[0], ref_feat.shape[1]
        
        # Normalize
        target_feat_norm = target_feat / target_feat.norm(dim=-1, keepdim=True)
        ref_feat_norm = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat_norm = ref_feat_norm.permute(2, 0, 1).reshape(C, h * w)
        
        sim = target_feat_norm @ ref_feat_norm
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()

        # Point selection (Prior)
        def point_selection(mask_sim, topk=1):
            w, h = mask_sim.shape
            topk_xy = mask_sim.flatten(0).topk(topk)[1]
            topk_x = (topk_xy // h).unsqueeze(0)
            topk_y = (topk_xy - topk_x * h)
            topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
            topk_label = np.array([1] * topk)
            topk_xy = topk_xy.cpu().numpy()
            return topk_xy, topk_label

        topk_xy, topk_label = point_selection(sim, topk=1)

        for train_idx in range(args.train_epoch_inside):
            masks, scores, logits, logits_high = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    multimask_output=True)
            
            logits_high = logits_high.flatten(1)
            
            # Weighted Sum
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            logits_high = logits_high * weights
            logits_high = logits_high.sum(0).unsqueeze(0)
            
            dice_loss = calculate_dice_loss(logits_high, gt_mask)
            focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
            loss = dice_loss + focal_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            final_loss = loss.item()
            
        # Check if this is the best model
        # We prefer lower loss
        if final_loss < best_loss:
            best_loss = final_loss
            best_ref_idx = idx
            best_weights_state = mask_weights.state_dict() # Save CPU state?
            # We must clone target_feat
            best_target_feat = target_feat.clone()
            best_ref_image_path = str(ref_image_file)
            
            print(f"Update Best: Idx {idx} | Loss {best_loss:.4f} | {name}")
            
        valid_count += 1

    # End of Loop
    print(f"\nTraining Finished. Valid Scans: {valid_count}.")
    print(f"Best Reference: {best_ref_image_path}")
    print(f"Best Loss: {best_loss:.4f}")
    
    # Save Feature and Weights
    if best_target_feat is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "target_feat": best_target_feat.cpu(),
            "mask_weights": best_weights_state, # State dict
            "ref_image_path": best_ref_image_path,
            "best_loss": best_loss,
            "sam_type": args.sam_type
        }
        
        torch.save(save_dict, str(save_path))
        print(f"Model saved to {save_path}")
    else:
        print("Failed to train any valid model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Save PerSAM-F Custom Model")
    parser.add_argument("--data", type=str, required=True, help="Path to formatted dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--sam_type", type=str, default="vit_t", help="SAM type")
    parser.add_argument("--save_path", type=str, default="weight/persam_f_stone.pt", help="Where to save .pt file")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train_epoch_inside", type=int, default=100)
    
    args = parser.parse_args()
    train_and_save(args)
