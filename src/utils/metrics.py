"""
Training Metrics Module

Provides comprehensive evaluation metrics for segmentation model training.
All metrics are designed to work with PyTorch tensors.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional

# Try importing optional dependencies
try:
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ==============================================================================
# Core Segmentation Metrics
# ==============================================================================

def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Intersection over Union (Jaccard Index).
    
    Args:
        pred: Predicted mask (B, C, H, W) or (B, H, W)
        target: Ground truth mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


def iou_per_class(pred: torch.Tensor, mask: torch.Tensor, num_classes: int = 2) -> Tuple[float, float]:
    """
    Calculate IoU for each class (background and foreground).
    
    Args:
        pred: Predicted mask
        mask: Ground truth mask
        num_classes: Number of classes (default 2 for binary)
        
    Returns:
        Tuple of (iou_background, iou_foreground)
    """
    pred = (pred > 0.5).float()
    mask = mask.float()
    iou_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        mask_cls = (mask == cls).float()
        intersection = (pred_cls * mask_cls).sum()
        union = pred_cls.sum() + mask_cls.sum() - intersection
        
        if union == 0:
            iou = torch.tensor(1.0)
        else:
            iou = intersection / union
        iou_scores.append(iou.item())
    
    return tuple(iou_scores)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice Coefficient (F1 Score for segmentation).
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate overall pixel accuracy.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        
    Returns:
        Pixel accuracy (correct pixels / total pixels)
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    correct = (pred == target).float().sum()
    total = target.numel()
    
    return correct / total


def precision(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate precision (positive predictive value).
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
        
    Returns:
        Precision score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    
    return (true_positive + smooth) / (predicted_positive + smooth)


def recall(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate recall (sensitivity, true positive rate).
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
        
    Returns:
        Recall score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    return (true_positive + smooth) / (actual_positive + smooth)


def f1_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
        
    Returns:
        F1 score
    """
    prec = precision(pred, target, smooth)
    rec = recall(pred, target, smooth)
    
    return 2 * (prec * rec) / (prec + rec + smooth)


def specificity(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate specificity (true negative rate).
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
        
    Returns:
        Specificity score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    true_negative = ((1 - pred) * (1 - target)).sum()
    actual_negative = (1 - target).sum()
    
    return (true_negative + smooth) / (actual_negative + smooth)


# ==============================================================================
# Image Quality Metrics
# ==============================================================================

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        
    Returns:
        SSIM score
    """
    if not HAS_SKIMAGE:
        return 0.0
        
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    
    data_range = pred_np.max() - pred_np.min()
    if data_range == 0:
        data_range = 1.0
        
    return ssim_func(pred_np, target_np, win_size=3, data_range=data_range)


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        
    Returns:
        PSNR value in dB
    """
    if not HAS_SKIMAGE:
        return 0.0
        
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    
    return psnr_func(target_np, pred_np)


def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Squared Error.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        
    Returns:
        MSE value
    """
    return ((pred - target) ** 2).mean()


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Absolute Error.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        
    Returns:
        MAE value
    """
    return (pred - target).abs().mean()


# ==============================================================================
# Boundary and Edge Metrics
# ==============================================================================

def boundary_iou(pred: torch.Tensor, target: torch.Tensor, dilation: int = 2) -> torch.Tensor:
    """
    Calculate Boundary IoU (focuses on edge accuracy).
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        dilation: Size of boundary region
        
    Returns:
        Boundary IoU score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    # Simple boundary extraction using max pooling
    kernel_size = 2 * dilation + 1
    
    # Get boundaries by comparing with dilated versions
    pred_dilated = torch.nn.functional.max_pool2d(
        pred.unsqueeze(0) if pred.dim() == 3 else pred,
        kernel_size=kernel_size, stride=1, padding=dilation
    )
    target_dilated = torch.nn.functional.max_pool2d(
        target.unsqueeze(0) if target.dim() == 3 else target,
        kernel_size=kernel_size, stride=1, padding=dilation
    )
    
    pred_boundary = (pred_dilated.squeeze() - pred.squeeze()).abs()
    target_boundary = (target_dilated.squeeze() - target.squeeze()).abs()
    
    # Calculate IoU on boundaries
    intersection = (pred_boundary * target_boundary).sum()
    union = pred_boundary.sum() + target_boundary.sum() - intersection
    
    return (intersection + 1e-6) / (union + 1e-6)


# ==============================================================================
# Comprehensive Metrics Calculator
# ==============================================================================

def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate all available metrics at once.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        
    Returns:
        Dictionary with all metric values
    """
    pred = pred.float()
    target = target.float()
    
    metrics = {
        'iou': iou_score(pred, target).item(),
        'dice': dice_coefficient(pred, target).item(),
        'pixel_accuracy': pixel_accuracy(pred, target).item(),
        'precision': precision(pred, target).item(),
        'recall': recall(pred, target).item(),
        'f1': f1_score(pred, target).item(),
        'specificity': specificity(pred, target).item(),
        'mse': mean_squared_error(pred, target).item(),
        'mae': mean_absolute_error(pred, target).item(),
    }
    
    # Add image quality metrics if available
    if HAS_SKIMAGE:
        metrics['ssim'] = calculate_ssim(pred, target)
        metrics['psnr'] = calculate_psnr(pred, target)
    
    # Add per-class IoU
    iou_bg, iou_fg = iou_per_class(pred, target)
    metrics['iou_background'] = iou_bg
    metrics['iou_foreground'] = iou_fg
    
    return metrics


# ==============================================================================
# Loss Functions (for training)
# ==============================================================================

class DiceLoss(torch.nn.Module):
    """Dice Loss for segmentation training."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        return 1 - dice_coefficient(pred, target, self.smooth)


class IoULoss(torch.nn.Module):
    """IoU (Jaccard) Loss for segmentation training."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        return 1 - iou_score(pred, target, self.smooth)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling class imbalance.
    Useful when foreground is much smaller than background.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Binary cross entropy
        bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal weight
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        return (focal_weight * bce).mean()


class CombinedLoss(torch.nn.Module):
    """
    Combined loss: BCE + Dice Loss.
    Often gives better results than either alone.
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    # Create example tensors
    pred = torch.rand(1, 1, 256, 256)
    target = (torch.rand(1, 1, 256, 256) > 0.5).float()
    
    # Calculate all metrics
    metrics = calculate_all_metrics(pred, target)
    
    print("Segmentation Metrics:")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
