"""
Data Loading Utilities

Unified module for all dataset classes and dataloader creation:
- SegmentationDataset: Basic NPY file loader
- RockSegmentationDataset: Rock segmentation with patch extraction
- SyntheticRockDataset: Synthetic rock image generator
- create_dataloaders: Factory function for train/val/test loaders
"""

import os
import glob
import random
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Try importing augmentation library
try:
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARNING] albumentations not installed. Using basic transforms.")


# ==============================================================================
# Default Transforms
# ==============================================================================

def get_train_transforms(use_augmentation: bool = True):
    """Get training transforms with augmentation."""
    if not HAS_ALBUMENTATIONS:
        return None
    
    if use_augmentation:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GridDistortion(p=0.1),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_val_transforms():
    """Get validation/test transforms (no augmentation)."""
    if not HAS_ALBUMENTATIONS:
        return None
    
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ==============================================================================
# Dataset Classes
# ==============================================================================

class SegmentationDataset(Dataset):
    """
    Basic segmentation dataset for pre-saved NPY files.
    
    Expects:
        - image_folder: Directory with .npy image files
        - mask_folder: Directory with .npy mask files
    """
    
    def __init__(self, image_folder: str, mask_folder: str, transform=None):
        self.image_files = sorted(glob.glob(os.path.join(image_folder, '*.npy')))
        self.mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.npy')))
        self.transform = transform
        
        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        
        print(f"[SegmentationDataset] Loaded {len(self.image_files)} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])
        
        if self.transform and HAS_ALBUMENTATIONS:
            # Albumentations expects HWC format
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            transformed = self.transform(image=image, mask=mask.astype(np.float32))
            image = transformed['image']
            mask = transformed['mask'].unsqueeze(0)
        else:
            # Manual tensor conversion
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
            image = image.repeat(3, 1, 1)  # Make 3-channel
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask


class RockSegmentationDataset(Dataset):
    """
    Rock segmentation dataset with automatic patch extraction.
    
    Features:
        - Extracts patches from full-size images
        - Caches patches as NPY files for faster loading
        - Filters empty patches (no rock content)
        - Supports augmentation via albumentations
    """
    
    PATCH_SAVE_DIR = "./npy_patches"
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        is_train: bool = True,
        patch_size: int = 256,
        cache_patches: bool = True,
        indices: Optional[List[int]] = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        self.patch_size = patch_size
        self.cache_patches = cache_patches
        self.paired_patches = []
        
        # Setup cache directories
        if cache_patches:
            os.makedirs(os.path.join(self.PATCH_SAVE_DIR, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.PATCH_SAVE_DIR, "masks"), exist_ok=True)
        
        # Try loading cached patches
        if self._load_cached_patches(indices):
            pass  # Loaded from cache
        else:
            self._extract_patches_from_images()
        
        # Set default transforms
        if self.transform is None:
            self.transform = get_train_transforms() if is_train else get_val_transforms()
        
        print(f"[RockSegmentationDataset] Initialized with {len(self.paired_patches)} patches")
    
    def _load_cached_patches(self, indices: Optional[List[int]]) -> bool:
        """Try to load pre-cached patches."""
        if not self.cache_patches:
            return False
        
        img_dir = os.path.join(self.PATCH_SAVE_DIR, "images")
        mask_dir = os.path.join(self.PATCH_SAVE_DIR, "masks")
        
        npy_images = sorted(glob.glob(os.path.join(img_dir, "img_*.npy")))
        npy_masks = sorted(glob.glob(os.path.join(mask_dir, "mask_*.npy")))
        
        if len(npy_images) == len(npy_masks) and len(npy_images) > 0:
            print(f"[INFO] Loading {len(npy_images)} cached patches from {self.PATCH_SAVE_DIR}")
            self.paired_patches = [
                (np.load(img), np.load(mask)) 
                for img, mask in zip(npy_images, npy_masks)
            ]
            if indices is not None:
                self.paired_patches = [self.paired_patches[i] for i in indices if i < len(self.paired_patches)]
            return True
        return False
    
    def _extract_patches_from_images(self):
        """Extract patches from raw images."""
        print(f"[INFO] Extracting patches from {self.image_dir}")
        
        img_cache_dir = os.path.join(self.PATCH_SAVE_DIR, "images")
        mask_cache_dir = os.path.join(self.PATCH_SAVE_DIR, "masks")
        
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG']:
            for img_path in glob.glob(os.path.join(self.image_dir, ext)):
                img_name = os.path.basename(img_path)
                if '_mask' in img_name:
                    continue
                
                # Find corresponding mask
                base = os.path.splitext(img_name)[0]
                mask_path = None
                for mask_ext in ['_mask.png', '_mask.PNG', '.png', '.PNG']:
                    p = os.path.join(self.mask_dir, f"{base}{mask_ext}")
                    if os.path.exists(p):
                        mask_path = p
                        break
                
                if mask_path is None:
                    continue
                
                # Load image and mask
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Extract patches
                h, w = image.shape[:2]
                ps = self.patch_size
                
                for y in range(0, h - ps + 1, ps):
                    for x in range(0, w - ps + 1, ps):
                        patch_img = image[y:y+ps, x:x+ps]
                        patch_mask = mask[y:y+ps, x:x+ps]
                        
                        # Skip if wrong size
                        if patch_img.shape[:2] != (ps, ps):
                            continue
                        
                        # Skip empty patches (no rocks)
                        if np.sum(patch_mask > 127) < 10:
                            continue
                        
                        # Cache if enabled
                        if self.cache_patches:
                            patch_id = len(self.paired_patches)
                            np.save(os.path.join(img_cache_dir, f"img_{patch_id}.npy"), patch_img)
                            np.save(os.path.join(mask_cache_dir, f"mask_{patch_id}.npy"), patch_mask)
                        
                        self.paired_patches.append((patch_img, patch_mask))
    
    def __len__(self):
        return len(self.paired_patches)
    
    def __getitem__(self, idx):
        image, mask = self.paired_patches[idx]
        mask = (mask > 127).astype(np.float32)
        
        if self.transform and HAS_ALBUMENTATIONS:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Manual conversion
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32)
        
        # Ensure mask has channel dimension
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = np.expand_dims(mask, 0)
        
        return image, mask


class SyntheticRockDataset(Dataset):
    """
    Generates synthetic rock-like images for training without real data.
    
    Useful for:
        - Initial model training
        - Data augmentation
        - Testing pipelines
    """
    
    # Rock color palette
    ROCK_COLORS = [
        (200, 190, 170),  # Beige
        (150, 150, 150),  # Gray
        (100, 100, 100),  # Dark gray
        (180, 170, 150),  # Light beige
        (120, 110, 100),  # Brown-gray
    ]
    
    BACKGROUND_COLORS = [
        (220, 210, 200),  # Light beige
        (180, 180, 180),  # Light gray
        (160, 150, 140),  # Tan
    ]
    
    def __init__(
        self,
        num_samples: int = 5000,
        patch_size: int = 128,
        transform=None,
        num_rocks_range: Tuple[int, int] = (5, 30),
        rock_size_range: Tuple[int, int] = (10, 80)
    ):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.transform = transform or get_train_transforms()
        self.num_rocks_range = num_rocks_range
        self.rock_size_range = rock_size_range
        
        print(f"[SyntheticRockDataset] Created with {num_samples} samples ({patch_size}x{patch_size})")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image, mask = self._generate_rock_image()
        
        if self.transform and HAS_ALBUMENTATIONS:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].unsqueeze(0)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask
    
    def _generate_rock_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic rock image with mask."""
        ps = self.patch_size
        image = np.zeros((ps, ps, 3), dtype=np.uint8)
        mask = np.zeros((ps, ps), dtype=np.float32)
        
        # Fill background
        bg_color = random.choice(self.BACKGROUND_COLORS)
        image[:, :] = bg_color
        
        # Generate rocks
        num_rocks = random.randint(*self.num_rocks_range)
        
        for _ in range(num_rocks):
            rock_size = random.randint(*self.rock_size_range)
            x = random.randint(0, ps - rock_size)
            y = random.randint(0, ps - rock_size)
            rock_color = random.choice(self.ROCK_COLORS)
            
            # Generate irregular polygon
            points = self._generate_rock_polygon(x, y, rock_size)
            
            # Draw rock
            cv2.fillPoly(image, [points], rock_color)
            cv2.fillPoly(mask, [points], 1.0)
            
            # Add texture
            self._add_rock_texture(image, mask, x, y, rock_size, rock_color)
        
        return image, mask
    
    def _generate_rock_polygon(self, x: int, y: int, size: int) -> np.ndarray:
        """Generate irregular polygon points for a rock shape."""
        num_points = random.randint(5, 8)
        points = []
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            noise = random.uniform(0.5, 1.2)
            px = int(x + size/2 + size/2 * noise * np.cos(angle))
            py = int(y + size/2 + size/2 * noise * np.sin(angle))
            px = max(0, min(px, self.patch_size - 1))
            py = max(0, min(py, self.patch_size - 1))
            points.append([px, py])
        
        return np.array(points, dtype=np.int32)
    
    def _add_rock_texture(self, image, mask, x, y, size, base_color):
        """Add texture details to a rock."""
        for _ in range(random.randint(1, 5)):
            tex_size = random.randint(2, 8)
            tx = random.randint(x, x + size - tex_size)
            ty = random.randint(y, y + size - tex_size)
            
            if 0 <= tx < self.patch_size - tex_size and 0 <= ty < self.patch_size - tex_size:
                if mask[ty + tex_size // 2, tx + tex_size // 2] > 0.5:
                    texture_color = tuple(
                        max(0, min(255, c + random.randint(-30, 30)))
                        for c in base_color
                    )
                    cv2.rectangle(image, (tx, ty), (tx + tex_size, ty + tex_size), texture_color, -1)


# ==============================================================================
# DataLoader Factory
# ==============================================================================

def create_dataloaders(
    batch_size: int = 16,
    num_train: int = 5000,
    num_val: int = 500,
    num_test: int = 500,
    patch_size: int = 256,
    image_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_synthetic: bool = False,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        batch_size: Batch size for all loaders
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
        patch_size: Size of image patches
        image_dir: Directory with images (None for synthetic)
        mask_dir: Directory with masks (None for synthetic)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        use_synthetic: Force use of synthetic data
        verbose: Print progress messages
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    if verbose:
        print(f"[create_dataloaders] batch={batch_size}, train={num_train}, val={num_val}, test={num_test}")
    
    if image_dir and mask_dir and not use_synthetic:
        # Use real data
        full_dataset = RockSegmentationDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            transform=None,
            patch_size=patch_size
        )
        
        # Calculate split sizes
        total = len(full_dataset)
        train_size = min(num_train, int(0.8 * total))
        val_size = min(num_val, int(0.1 * total))
        test_size = min(num_test, total - train_size - val_size)
        
        # Create subsets
        indices = list(range(total))
        train_dataset = Subset(full_dataset, indices[:train_size])
        val_dataset = Subset(full_dataset, indices[train_size:train_size + val_size])
        test_dataset = Subset(full_dataset, indices[train_size + val_size:train_size + val_size + test_size])
        
        # Set transforms (affects underlying dataset)
        full_dataset.transform = train_transform
        
    else:
        # Use synthetic data
        if verbose:
            print("[INFO] Using synthetic rock dataset")
        
        train_dataset = SyntheticRockDataset(
            num_samples=num_train, patch_size=patch_size, transform=train_transform
        )
        val_dataset = SyntheticRockDataset(
            num_samples=num_val, patch_size=patch_size, transform=val_transform
        )
        test_dataset = SyntheticRockDataset(
            num_samples=num_test, patch_size=patch_size, transform=val_transform
        )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


# ==============================================================================
# Raw File Loading (for Digital Rocks Portal data)
# ==============================================================================

def load_raw_file(filename: str, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Load raw binary file into numpy array.
    
    Used for loading Digital Rocks Portal volumetric data.
    
    Args:
        filename: Path to .raw file
        shape: Expected shape (depth, height, width)
        
    Returns:
        Numpy array with the loaded data
    """
    return np.fromfile(filename, dtype=np.uint8).reshape(shape)


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    print("Testing DataLoader utilities...")
    
    # Test synthetic dataset
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=4, num_train=100, num_val=20, num_test=20,
        patch_size=128, use_synthetic=True
    )
    
    # Get one batch
    for images, masks in train_loader:
        print(f"Batch - Images: {images.shape}, Masks: {masks.shape}")
        break
    
    print("✅ DataLoader test passed!")


def download_file(url, filename):
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        total_size = int(response.getheader('Content-Length').strip())
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
            while True:
                data = response.read(1024)
                if not data:
                    break
                out_file.write(data)
                pbar.update(len(data))