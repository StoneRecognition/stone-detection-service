"""
Training Visualization Module

Provides visualization utilities for model training:
- Prediction comparisons (input, ground truth, prediction)
- Training metrics plots
- Training progress visualization

Uses the metrics module for consistent metric calculation.
"""

import os
import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import metrics from local module
from .metrics import (
    iou_score, dice_coefficient, pixel_accuracy,
    precision, recall, calculate_all_metrics
)


def show_predictions(
    model: torch.nn.Module,
    dataloader,
    epoch: int,
    device: torch.device,
    num_examples: int = 5,
    save_dir: str = 'results'
) -> None:
    """
    Visualize model predictions with metrics overlay.
    
    Args:
        model: Trained model
        dataloader: DataLoader with validation data
        epoch: Current epoch number
        device: Device (cuda/cpu)
        num_examples: Number of examples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    images_shown = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            outputs_sigmoid = torch.sigmoid(outputs)
            outputs_binary = (outputs_sigmoid > 0.5).float()
            
            # Move to CPU for visualization
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            outputs_np = outputs_binary.cpu().numpy()
            
            # Denormalization parameters
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            for i in range(images_np.shape[0]):
                if images_shown >= num_examples:
                    _cleanup_memory(images, masks, outputs)
                    return
                
                # Calculate metrics for this sample
                pred_tensor = outputs_binary[i:i+1]
                mask_tensor = masks[i:i+1]
                
                sample_iou = iou_score(pred_tensor, mask_tensor).item()
                sample_dice = dice_coefficient(pred_tensor, mask_tensor).item()
                sample_acc = pixel_accuracy(pred_tensor, mask_tensor).item()
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Input image
                img = images_np[i]
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    axes[0].imshow(img)
                else:
                    axes[0].imshow(img.squeeze(), cmap='gray')
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                # Ground truth mask
                axes[1].imshow(masks_np[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Predicted mask with metrics
                axes[2].imshow(outputs_np[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[2].set_title(f'Prediction\nIoU: {sample_iou:.3f} | Dice: {sample_dice:.3f} | Acc: {sample_acc:.3f}')
                axes[2].axis('off')
                
                plt.suptitle(f'Epoch {epoch} - Sample {images_shown + 1}', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'Prediction_Epoch_{epoch}_Sample_{images_shown + 1}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                images_shown += 1
            
            _cleanup_memory(images, masks, outputs)


def plot_metrics(
    train_metrics: Dict[str, List[float]],
    eval_metrics: Dict[str, List[float]],
    save_dir: str = 'metrics'
) -> None:
    """
    Plot training and evaluation metrics over epochs.
    
    Args:
        train_metrics: Dictionary of training metrics per epoch
        eval_metrics: Dictionary of evaluation metrics per epoch
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    # Create comprehensive metrics plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_metrics['loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, eval_metrics['loss'], 'r-', label='Eval', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IoU
    ax = axes[0, 1]
    ax.plot(epochs, train_metrics['iou'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, eval_metrics['iou'], 'r-', label='Eval', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Intersection over Union')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dice Coefficient
    ax = axes[0, 2]
    ax.plot(epochs, train_metrics['dice'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, eval_metrics['dice'], 'r-', label='Eval', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_title('Dice Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-class IoU
    ax = axes[1, 0]
    if 'iou_bg' in train_metrics and 'iou_fg' in train_metrics:
        ax.plot(epochs, train_metrics['iou_bg'], 'b--', label='Train BG', linewidth=2)
        ax.plot(epochs, train_metrics['iou_fg'], 'b-', label='Train FG', linewidth=2)
        ax.plot(epochs, eval_metrics['iou_bg'], 'r--', label='Eval BG', linewidth=2)
        ax.plot(epochs, eval_metrics['iou_fg'], 'r-', label='Eval FG', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SSIM
    ax = axes[1, 1]
    if 'ssim' in train_metrics:
        ax.plot(epochs, train_metrics['ssim'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, eval_metrics['ssim'], 'r-', label='Eval', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('Structural Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PSNR
    ax = axes[1, 2]
    if 'psnr' in train_metrics:
        ax.plot(epochs, train_metrics['psnr'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, eval_metrics['psnr'], 'r-', label='Eval', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Peak Signal-to-Noise Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create individual metric plots
    _plot_individual_metrics(train_metrics, eval_metrics, save_dir, epochs)


def _plot_individual_metrics(
    train_metrics: Dict[str, List[float]],
    eval_metrics: Dict[str, List[float]],
    save_dir: str,
    epochs: range
) -> None:
    """Create individual plots for each metric."""
    
    for metric_name in train_metrics.keys():
        if metric_name not in eval_metrics:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_metrics[metric_name], 'b-', label='Train', linewidth=2)
        plt.plot(epochs, eval_metrics[metric_name], 'r-', label='Eval', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_name.upper(), fontsize=12)
        plt.title(f'{metric_name.upper()} over Training', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def plot_learning_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = 'learning_curve.png'
) -> None:
    """
    Plot learning curves for loss.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_loss = min(val_losses)
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.scatter([best_epoch], [best_loss], color='g', s=100, zorder=5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Learning Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_batch_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    save_path: str,
    max_samples: int = 8
) -> None:
    """
    Visualize a batch of predictions in a grid.
    
    Args:
        images: Input images (B, C, H, W)
        masks: Ground truth masks (B, 1, H, W)
        predictions: Predicted masks (B, 1, H, W)
        save_path: Path to save the visualization
        max_samples: Maximum number of samples to show
    """
    batch_size = min(images.shape[0], max_samples)
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(batch_size):
        # Image
        img = images[i].cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1)
            axes[i, 0].imshow(img)
        else:
            axes[i, 0].imshow(img.squeeze(), cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # Mask
        axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = (predictions[i] > 0.5).float()
        axes[i, 2].imshow(pred.cpu().squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _cleanup_memory(*tensors) -> None:
    """Clean up GPU memory after visualization."""
    for tensor in tensors:
        if tensor is not None:
            del tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
