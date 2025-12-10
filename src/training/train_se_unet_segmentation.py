#!/usr/bin/env python3
"""
Train SE-UNet for Rock Segmentation

This script trains an SE-UNet (Squeeze-and-Excitation UNet) model on rock segmentation data.
Uses centralized utilities from src/utils/.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import from new structure
from src.models.se_unet import SE_PP_UNet
from src.utils.dataloader import RockSegmentationDataset, create_dataloaders
from src.utils.metrics import dice_coefficient, iou_per_class, calculate_ssim, calculate_psnr
from src.utils.checkpoint_utils import save_checkpoint
from src.utils.training_visualization import show_predictions, plot_metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    epoch_loss = 0
    dice_scores = []
    iou_scores = []
    ssim_scores = []
    psnr_scores = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            epoch_loss += loss.item()

            preds = (outputs > 0.5).float()
            dice_scores.append(dice_coefficient(preds, masks).item())
            iou_scores.append(iou_per_class(preds, masks))
            ssim_scores.append(calculate_ssim(preds, masks))
            psnr_scores.append(calculate_psnr(preds, masks))

    return {
        "loss": epoch_loss / len(dataloader),
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores),
        "ssim": np.mean(ssim_scores),
        "psnr": np.mean(psnr_scores)
    }


def early_stopping(patience, best_loss, counter, current_loss):
    """Check for early stopping condition."""
    if current_loss < best_loss:
        return current_loss, 0, True
    else:
        counter += 1
        if counter >= patience:
            return best_loss, counter, False
        return best_loss, counter, True


def train_model(model, train_loader, eval_loader, optimizer, device,
                num_epochs, pos_weight, patience, save_dir):
    """Train the model with early stopping."""
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    best_loss = float('inf')
    stop_counter = 0
    keep_training = True

    train_metrics = {'loss': [], 'iou': [], 'dice': [], 'ssim': [], 'psnr': []}
    eval_metrics = {'loss': [], 'iou': [], 'dice': [], 'ssim': [], 'psnr': []}
    
    print(f"Starting training for {num_epochs} epochs with patience {patience}")
    print(f"Using device: {device}")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Saving checkpoints to {save_dir}")

    for epoch in range(1, num_epochs+1):
        if not keep_training:
            print(f"Early stopping at epoch {epoch-1}")
            break

        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_ssim = 0.0
        running_psnr = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_iou += np.mean(iou_per_class(preds, masks))
            running_dice += dice_coefficient(preds, masks).item()
            running_ssim += calculate_ssim(preds, masks)
            running_psnr += calculate_psnr(preds, masks)

        train_loss = running_loss / len(train_loader)
        train_iou = running_iou / len(train_loader)
        train_dice = running_dice / len(train_loader)
        train_metrics['loss'].append(train_loss)
        train_metrics['iou'].append(train_iou)
        train_metrics['dice'].append(train_dice)
        train_metrics['ssim'].append(running_ssim / len(train_loader))
        train_metrics['psnr'].append(running_psnr / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_ssim = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(eval_loader, desc=f"Epoch {epoch} - Val"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_iou += np.mean(iou_per_class(preds, masks))
                val_dice += dice_coefficient(preds, masks).item()
                val_ssim += calculate_ssim(preds, masks)
                val_psnr += calculate_psnr(preds, masks)

        val_loss /= len(eval_loader)
        val_iou /= len(eval_loader)
        val_dice /= len(eval_loader)
        eval_metrics['loss'].append(val_loss)
        eval_metrics['iou'].append(val_iou)
        eval_metrics['dice'].append(val_dice)
        eval_metrics['ssim'].append(val_ssim / len(eval_loader))
        eval_metrics['psnr'].append(val_psnr / len(eval_loader))

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val IoU={val_iou:.4f}, Val Dice={val_dice:.4f}")
        
        best_loss, stop_counter, keep_training = early_stopping(patience, best_loss, stop_counter, val_loss)
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        }, is_best=(val_loss == best_loss), folder=save_dir)

        # Visualize predictions periodically
        if epoch % 5 == 0 or epoch == num_epochs:
            pred_dir = Path(save_dir) / f"predictions_epoch_{epoch}"
            show_predictions(model, eval_loader, epoch, device, save_dir=str(pred_dir))

    # Plot final metrics
    plot_metrics(train_metrics, eval_metrics, save_dir)

    return model


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train SE-UNet for Rock Segmentation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--pos_weight', type=float, default=2.0, help='Positive class weight')
    parser.add_argument('--save_dir', type=str, default='checkpoints/se_unet', help='Checkpoint directory')
    parser.add_argument('--image_dir', type=str, default=None, help='Image directory')
    parser.add_argument('--mask_dir', type=str, default=None, help='Mask directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    if args.image_dir and args.mask_dir:
        train_loader, val_loader, _ = create_dataloaders(
            batch_size=args.batch_size,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            patch_size=256
        )
    else:
        train_loader, val_loader, _ = create_dataloaders(
            batch_size=args.batch_size,
            num_train=1000,
            num_val=200,
            patch_size=256,
            use_synthetic=True
        )

    # Initialize model
    model = SE_PP_UNet(in_channels=3, out_channels=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs,
        pos_weight=args.pos_weight,
        patience=args.patience,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
