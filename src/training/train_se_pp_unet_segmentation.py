#!/usr/bin/env python3
"""
Train SE+PPM-UNet for Rock Segmentation

This script trains an SE+PPM-UNet (Squeeze-and-Excitation + Pyramid Pooling Module UNet) 
model on rock segmentation data. Uses centralized utilities from src/utils/.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from new structure
from src.models.se_unet import SE_PP_UNet
from src.utils.dataloader import SegmentationDataset, create_dataloaders
from src.utils.metrics import iou_per_class, dice_coefficient, calculate_ssim, calculate_psnr
from src.utils.training_visualization import show_predictions, plot_metrics
from src.utils.checkpoint_utils import save_checkpoint

# Store metrics for plotting
train_metrics = {'loss': [], 'iou': [], 'iou_bg': [], 'iou_fg': [], 'dice': [], 'ssim': [], 'psnr': []}
eval_metrics = {'loss': [], 'iou': [], 'iou_bg': [], 'iou_fg': [], 'dice': [], 'ssim': [], 'psnr': []}


def count_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, input_size=(1, 3, 256, 256), device='cuda', num_iterations=100):
    """Measure the average inference time of the model."""
    model.eval()
    model = model.to(device)
    x = torch.randn(input_size).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure time
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    return avg_time * 1000  # Convert to milliseconds


def train_model(model, train_loader, eval_loader, optimizer, device, num_epochs=25, pos_weight=None):
    """Train the model with optional class balancing."""
    model = model.to(device)
    best_eval_loss = float('inf')
    
    # Loss function with class balancing
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        iou_score = 0.0
        iou_bg_score = 0.0
        iou_fg_score = 0.0
        dice_score, ssim_score, psnr_score = 0.0, 0.0, 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute metrics
            outputs_sigmoid = torch.sigmoid(outputs)
            iou_bg, iou_fg = iou_per_class(outputs_sigmoid, masks)
            iou_score += (iou_bg + iou_fg) / 2
            iou_bg_score += iou_bg
            iou_fg_score += iou_fg
            dice_score += dice_coefficient(outputs_sigmoid, masks).item()
            ssim_score += calculate_ssim(outputs_sigmoid, masks)
            psnr_score += calculate_psnr(outputs_sigmoid, masks)

        epoch_loss = running_loss / len(train_loader)
        epoch_iou = iou_score / len(train_loader)
        epoch_iou_bg = iou_bg_score / len(train_loader)
        epoch_iou_fg = iou_fg_score / len(train_loader)
        epoch_dice = dice_score / len(train_loader)
        epoch_ssim = ssim_score / len(train_loader)
        epoch_psnr = psnr_score / len(train_loader)

        # Save training metrics
        train_metrics['loss'].append(epoch_loss)
        train_metrics['iou'].append(epoch_iou)
        train_metrics['iou_bg'].append(epoch_iou_bg)
        train_metrics['iou_fg'].append(epoch_iou_fg)
        train_metrics['dice'].append(epoch_dice)
        train_metrics['ssim'].append(epoch_ssim)
        train_metrics['psnr'].append(epoch_psnr)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f} '
              f'(BG: {epoch_iou_bg:.4f}, FG: {epoch_iou_fg:.4f}), Dice: {epoch_dice:.4f}')

        # Evaluate
        eval_loss, eval_iou = evaluate_model(model, eval_loader, criterion, device, epoch+1)
        
        # Update learning rate
        scheduler.step(eval_iou)

        # Save best model
        is_best = eval_loss < best_eval_loss
        if is_best:
            best_eval_loss = eval_loss
            print(f"New best model at epoch {epoch+1} with loss: {best_eval_loss:.4f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': eval_loss
        }, is_best)

    # Final plots
    plot_metrics(train_metrics, eval_metrics)
    return model


def evaluate_model(model, dataloader, criterion, device, epoch):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    iou_score = 0.0
    iou_bg_score = 0.0
    iou_fg_score = 0.0
    dice_score, ssim_score, psnr_score = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch} - Evaluation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            outputs_sigmoid = torch.sigmoid(outputs)
            iou_bg, iou_fg = iou_per_class(outputs_sigmoid, masks)
            iou_score += (iou_bg + iou_fg) / 2
            iou_bg_score += iou_bg
            iou_fg_score += iou_fg
            dice_score += dice_coefficient(outputs_sigmoid, masks).item()
            ssim_score += calculate_ssim(outputs_sigmoid, masks)
            psnr_score += calculate_psnr(outputs_sigmoid, masks)

    # Calculate averages
    avg_loss = running_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_iou_bg = iou_bg_score / len(dataloader)
    avg_iou_fg = iou_fg_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    avg_ssim = ssim_score / len(dataloader)
    avg_psnr = psnr_score / len(dataloader)

    # Save eval metrics
    eval_metrics['loss'].append(avg_loss)
    eval_metrics['iou'].append(avg_iou)
    eval_metrics['iou_bg'].append(avg_iou_bg)
    eval_metrics['iou_fg'].append(avg_iou_fg)
    eval_metrics['dice'].append(avg_dice)
    eval_metrics['ssim'].append(avg_ssim)
    eval_metrics['psnr'].append(avg_psnr)

    print(f'Eval - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f} '
          f'(BG: {avg_iou_bg:.4f}, FG: {avg_iou_fg:.4f}), Dice: {avg_dice:.4f}')

    # Visualize periodically
    if epoch % 5 == 0:
        show_predictions(model, dataloader, epoch, device, num_examples=5)

    return avg_loss, avg_iou


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train SE+PPM-UNet for Rock Segmentation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pos_weight', type=float, default=2.0, help='Positive class weight')
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
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Measure inference time
    if device.type == 'cuda':
        inf_time = measure_inference_time(model, device=str(device))
        print(f"Inference time: {inf_time:.2f} ms")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train_model(model, train_loader, val_loader, optimizer, device, 
                num_epochs=args.epochs, pos_weight=args.pos_weight)


if __name__ == "__main__":
    main()
