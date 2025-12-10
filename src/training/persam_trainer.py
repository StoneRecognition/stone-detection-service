#!/usr/bin/env python3
"""
PerSAM-F Trainer Module

Personalized Segment Anything Model with Fine-tuning (PerSAM-F).
Fine-tunes SAM's mask decoder using a single reference mask in ~10 seconds.

Key features:
- Freezes image encoder and prompt encoder (preserves SAM's knowledge)
- Introduces 2 learnable mask weights for scale-aware aggregation
- Uses DICE Loss + Focal Loss for precise boundary optimization
- Completes training in approximately 10 seconds

Usage:
    from src.training.persam_trainer import PerSAMTrainer
    
    trainer = PerSAMTrainer(ref_image, ref_mask)
    trainer.train()
    trainer.save("weight/persam_stone.pth")
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load config
try:
    from src.utils.settings import config
    weights_dir = Path(config.get('paths.weights_dir', 'weights'))
    persam_config = {
        'learning_rate': config.get('persam.learning_rate', 0.0001),
        'iterations': config.get('persam.iterations', 1000),
        'sam_type': config.get('persam.sam_type', 'vit_h'),
    }
except ImportError:
    weights_dir = Path('./weight')
    persam_config = {
        'learning_rate': 0.0001,
        'iterations': 1000,
        'sam_type': 'vit_h',
    }


class DiceLoss(nn.Module):
    """
    DICE Loss for segmentation.
    
    Measures overlap between predicted and ground truth masks.
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DICE loss.
        
        Args:
            pred: Predicted mask logits (B, H, W) or (B, 1, H, W)
            target: Ground truth mask (B, H, W) or (B, 1, H, W)
            
        Returns:
            DICE loss value
        """
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        pred_prob = torch.sigmoid(pred)
        
        intersection = (pred_prob * target).sum(dim=1)
        union = pred_prob.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for handling class imbalance.
    
    Focuses learning on hard examples by down-weighting easy ones.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: Predicted mask logits
            target: Ground truth mask
            
        Returns:
            Focal loss value
        """
        pred = pred.flatten()
        target = target.flatten()
        
        prob = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class PerSAMTrainer:
    """
    PerSAM-F (Personalized SAM with Fine-tuning) Trainer.
    
    Specializes SAM for a specific object category using a single
    reference image-mask pair. Training completes in ~10 seconds.
    
    Architecture:
        - Frozen: Image Encoder (ViT), Prompt Encoder
        - Trainable: 2 learnable mask weights for scale selection
    
    Attributes:
        model: SAM model instance
        ref_image: Reference image tensor
        ref_mask: Reference mask tensor
        mask_weights: Learnable parameters (2 weights)
    """
    
    SAM_TYPES = {
        'vit_h': 'sam_vit_h.pt',
        'vit_l': 'sam_vit_l.pt',
        'vit_b': 'sam_vit_b.pt',
        'vit_t': 'mobile_sam.pt',
    }
    
    def __init__(
        self,
        ref_image: Union[np.ndarray, torch.Tensor],
        ref_mask: Union[np.ndarray, torch.Tensor],
        sam_checkpoint: Optional[str] = None,
        sam_type: str = 'vit_h',
        device: Optional[str] = None,
        learning_rate: float = 0.0001,
        iterations: int = 1000,
    ):
        """
        Initialize PerSAM-F trainer.
        
        Args:
            ref_image: Reference RGB image (H, W, 3) or (3, H, W)
            ref_mask: Reference binary mask (H, W)
            sam_checkpoint: Path to SAM weights
            sam_type: SAM variant ('vit_h', 'vit_l', 'vit_b', 'vit_t')
            device: Computation device
            learning_rate: Learning rate for optimizer
            iterations: Number of training iterations
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"PerSAM-F Trainer using device: {self.device}")
        
        self.sam_type = sam_type
        self.learning_rate = learning_rate
        self.iterations = iterations
        
        # Set checkpoint path
        if sam_checkpoint is None:
            sam_file = self.SAM_TYPES.get(sam_type, 'sam_vit_h.pt')
            for check_dir in [weights_dir, project_root / 'weight']:
                check_path = check_dir / sam_file
                if check_path.exists():
                    sam_checkpoint = str(check_path)
                    break
            if sam_checkpoint is None:
                raise FileNotFoundError(f"SAM weights not found")
        
        self.sam_checkpoint = sam_checkpoint
        
        # Load SAM model
        self.model = self._load_sam()
        
        # Freeze all parameters
        self._freeze_model()
        
        # Initialize learnable mask weights
        self.mask_weights = self._init_learnable_weights()
        
        # Prepare reference data
        self.ref_image, self.ref_mask = self._prepare_reference(
            ref_image, ref_mask
        )
        
        # Pre-compute reference features
        self.ref_features = None
        self.ref_embedding = None
        
        # Training state
        self.is_trained = False
        self.training_log = []
    
    def _load_sam(self):
        """Load SAM model."""
        if self.sam_type == 'vit_t':
            from mobile_sam import sam_model_registry
            model_type = 'vit_t'
        else:
            from segment_anything import sam_model_registry
            model_type = self.sam_type
        
        model = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        model.to(device=self.device)
        logger.info(f"Loaded SAM ({self.sam_type}) from {self.sam_checkpoint}")
        return model
    
    def _freeze_model(self):
        """Freeze all SAM parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Froze all SAM parameters")
    
    def _init_learnable_weights(self) -> nn.Parameter:
        """
        Initialize learnable mask weights for scale selection.
        
        SAM outputs 3 masks at different scales (whole, part, subpart).
        These 2 weights learn the optimal aggregation for our target object.
        
        Returns:
            nn.Parameter with learnable weights
        """
        # Initialize weights for 3 SAM output masks
        # We use 2 learnable weights that combine to 3 via softmax
        weights = torch.ones(3, device=self.device) / 3.0
        mask_weights = nn.Parameter(weights.clone())
        mask_weights.requires_grad = True
        
        logger.info("Initialized 3 learnable mask weights (scale selection)")
        return mask_weights
    
    def _prepare_reference(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare reference image and mask tensors."""
        # Convert image
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:  # HWC -> CHW
                image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        image = image.to(self.device)
        
        # Convert mask
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        mask = mask.to(self.device)
        
        return image, mask
    
    def _get_target_embedding(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Get target location embedding from mask.
        
        Extracts the center point of the mask for prompt encoding.
        """
        # Get mask centroid
        mask_np = mask.squeeze().cpu().numpy()
        y_indices, x_indices = np.where(mask_np > 0.5)
        
        if len(y_indices) == 0:
            # Fallback to center
            h, w = mask_np.shape
            center_point = np.array([[w // 2, h // 2]])
        else:
            center_x = int(x_indices.mean())
            center_y = int(y_indices.mean())
            center_point = np.array([[center_x, center_y]])
        
        center_point = torch.from_numpy(center_point).float().to(self.device)
        center_label = torch.ones(1, 1, device=self.device)
        
        return center_point.unsqueeze(0), center_label
    
    def _compute_image_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """Compute image embedding using frozen encoder."""
        with torch.no_grad():
            image_embedding = self.model.image_encoder(image)
        return image_embedding
    
    def _forward(
        self,
        image_embedding: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through SAM with learnable weight aggregation.
        
        Returns:
            Aggregated mask prediction
        """
        # Get prompt embeddings (frozen)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
        
        # Get multi-scale mask predictions from decoder
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        
        # Apply learnable weights to aggregate masks
        # Softmax ensures weights sum to 1
        weights = F.softmax(self.mask_weights, dim=0)
        
        # Weighted sum of masks
        aggregated_mask = torch.zeros_like(low_res_masks[:, 0])
        for i in range(3):
            aggregated_mask += weights[i] * low_res_masks[:, i]
        
        # Upscale to original resolution
        upscaled_mask = F.interpolate(
            aggregated_mask.unsqueeze(1),
            size=self.ref_mask.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        
        return upscaled_mask
    
    def train(
        self,
        iterations: Optional[int] = None,
        learning_rate: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Fine-tune PerSAM-F using the reference mask.
        
        Only updates the 2 learnable mask weights while keeping
        the entire SAM model frozen.
        
        Args:
            iterations: Override default iteration count
            learning_rate: Override default learning rate
            verbose: Print progress every 100 iterations
            
        Returns:
            Training log dictionary
        """
        iterations = iterations or self.iterations
        learning_rate = learning_rate or self.learning_rate
        
        # Setup losses
        dice_loss = DiceLoss()
        focal_loss = SigmoidFocalLoss()
        
        # Setup optimizer (only for learnable weights)
        optimizer = AdamW([self.mask_weights], lr=learning_rate)
        
        # Pre-compute image embedding (frozen, only computed once)
        logger.info("Computing image embedding...")
        with torch.no_grad():
            # Prepare image for SAM encoder
            from segment_anything.utils.transforms import ResizeLongestSide
            transform = ResizeLongestSide(self.model.image_encoder.img_size)
            
            ref_image_np = self.ref_image.squeeze().cpu().numpy()
            if ref_image_np.shape[0] == 3:  # CHW -> HWC
                ref_image_np = ref_image_np.transpose(1, 2, 0)
            ref_image_np = ref_image_np.astype(np.uint8)
            
            input_image = transform.apply_image(ref_image_np)
            input_image_torch = torch.from_numpy(input_image).permute(2, 0, 1)
            input_image_torch = input_image_torch.unsqueeze(0).float().to(self.device)
            
            # Normalize
            pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(self.device)
            pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(self.device)
            input_image_torch = (input_image_torch - pixel_mean) / pixel_std
            
            # Pad to square
            h, w = input_image_torch.shape[-2:]
            padh = self.model.image_encoder.img_size - h
            padw = self.model.image_encoder.img_size - w
            input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
            
            image_embedding = self.model.image_encoder(input_image_torch)
        
        # Get point prompt from mask center
        point_coords, point_labels = self._get_target_embedding(self.ref_mask)
        
        # Scale point coordinates
        original_size = (ref_image_np.shape[0], ref_image_np.shape[1])
        point_coords_scaled = transform.apply_coords(
            point_coords.squeeze(0).cpu().numpy(),
            original_size
        )
        point_coords_scaled = torch.from_numpy(point_coords_scaled).unsqueeze(0).float().to(self.device)
        
        logger.info(f"Starting PerSAM-F training for {iterations} iterations...")
        start_time = time.time()
        
        self.training_log = []
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            pred_mask = self._forward(
                image_embedding,
                point_coords_scaled,
                point_labels,
            )
            
            # Compute losses
            d_loss = dice_loss(pred_mask, self.ref_mask)
            f_loss = focal_loss(pred_mask, self.ref_mask)
            total_loss = d_loss + f_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Log progress
            log_entry = {
                'iteration': i + 1,
                'dice_loss': d_loss.item(),
                'focal_loss': f_loss.item(),
                'total_loss': total_loss.item(),
            }
            self.training_log.append(log_entry)
            
            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Iter {i+1}/{iterations} | "
                    f"Dice: {d_loss.item():.4f} | "
                    f"Focal: {f_loss.item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f} seconds")
        
        # Log final weights
        final_weights = F.softmax(self.mask_weights, dim=0).detach().cpu().numpy()
        logger.info(f"Learned mask weights: {final_weights}")
        
        self.is_trained = True
        
        return {
            'total_time': total_time,
            'iterations': iterations,
            'final_loss': self.training_log[-1]['total_loss'],
            'final_weights': final_weights.tolist(),
        }
    
    def save(self, save_path: Union[str, Path]):
        """
        Save trained weights.
        
        Only saves the learnable mask weights, not the full SAM model.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'mask_weights': self.mask_weights.detach().cpu(),
            'sam_type': self.sam_type,
            'sam_checkpoint': self.sam_checkpoint,
            'training_log': self.training_log,
            'is_trained': self.is_trained,
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved PerSAM-F weights to: {save_path}")
    
    @classmethod
    def load(
        cls,
        weights_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> 'PerSAMTrainer':
        """
        Load trained PerSAM-F weights.
        
        Note: Requires a dummy reference to initialize, but weights
        will be overwritten by loaded checkpoint.
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Create dummy trainer (weights will be overwritten)
        dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy_mask = np.zeros((256, 256), dtype=np.float32)
        
        trainer = cls(
            ref_image=dummy_image,
            ref_mask=dummy_mask,
            sam_type=checkpoint['sam_type'],
            device=device,
        )
        
        # Load trained weights
        trainer.mask_weights = nn.Parameter(
            checkpoint['mask_weights'].to(trainer.device)
        )
        trainer.training_log = checkpoint.get('training_log', [])
        trainer.is_trained = checkpoint.get('is_trained', True)
        
        logger.info(f"Loaded PerSAM-F weights from: {weights_path}")
        return trainer


def train_persam(
    ref_image_path: Union[str, Path],
    ref_mask_path: Union[str, Path],
    output_path: Union[str, Path],
    sam_type: str = 'vit_h',
    iterations: int = 1000,
    learning_rate: float = 0.0001,
) -> Dict:
    """
    Convenience function to train PerSAM-F from file paths.
    
    Args:
        ref_image_path: Path to reference image
        ref_mask_path: Path to reference mask
        output_path: Path to save trained weights
        sam_type: SAM variant to use
        iterations: Training iterations
        learning_rate: Learning rate
        
    Returns:
        Training results dictionary
    """
    # Load reference data
    ref_image = cv2.imread(str(ref_image_path))
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    
    ref_mask = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
    ref_mask = (ref_mask > 127).astype(np.float32)
    
    # Create trainer and train
    trainer = PerSAMTrainer(
        ref_image=ref_image,
        ref_mask=ref_mask,
        sam_type=sam_type,
        iterations=iterations,
        learning_rate=learning_rate,
    )
    
    results = trainer.train()
    trainer.save(output_path)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PerSAM-F model")
    parser.add_argument("--ref-image", type=str, required=True,
                        help="Path to reference image")
    parser.add_argument("--ref-mask", type=str, required=True,
                        help="Path to reference mask")
    parser.add_argument("--output", "-o", type=str, 
                        default="weights/persam_stone.pth",
                        help="Output path for trained weights")
    parser.add_argument("--sam-type", type=str, default="vit_h",
                        choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
                        help="SAM model variant")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PerSAM-F Training")
    print("=" * 60)
    print(f"Reference image: {args.ref_image}")
    print(f"Reference mask: {args.ref_mask}")
    print(f"SAM type: {args.sam_type}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60)
    
    results = train_persam(
        ref_image_path=args.ref_image,
        ref_mask_path=args.ref_mask,
        output_path=args.output,
        sam_type=args.sam_type,
        iterations=args.iterations,
        learning_rate=args.lr,
    )
    
    print("\nTraining Results:")
    print(f"  Time: {results['total_time']:.1f} seconds")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Learned weights: {results['final_weights']}")
    print(f"\nWeights saved to: {args.output}")
