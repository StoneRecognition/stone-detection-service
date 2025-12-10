#!/usr/bin/env python3
"""
UNet Inference Script

Runs inference using trained SE-UNet model for rock segmentation.
Uses centralized configuration from src/utils/settings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
from glob import glob
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict

# Import from new structure
from src.models.se_unet import SE_PP_UNet

# Try loading config, use defaults if not available
try:
    from src.utils.settings import config
    MODEL_PATH = config.get('paths.weights', 'weight') + '/best_model.pth'
    IMAGE_DIR = config.get('paths.data', 'data') + '/raw'
    MASK_DIR = 'results/masks'
    OUT_DIR = 'results/contour_overlay'
except ImportError:
    MODEL_PATH = 'weight/best_model.pth'
    IMAGE_DIR = 'data/raw'
    MASK_DIR = 'results/masks'
    OUT_DIR = 'results/contour_overlay'

# Configuration
CONTOUR_COLOR = (0, 255, 0)  # Green in BGR
CONTOUR_THICKNESS = 2
N_CHANNELS = 3
N_CLASSES = 1
BASE_CHANNELS = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def load_image(path):
    img = Image.open(path).convert('RGB')
    w0, h0 = img.size
    new_w, new_h = (w0//16)*16, (h0//16)*16
    img_rs = img.resize((new_w, new_h), Image.BILINEAR)
    tensor = T.ToTensor()(img_rs).unsqueeze(0)
    return tensor, (w0, h0)

def save_mask(mask_tensor, orig_size, path):
    mask = mask_tensor.squeeze().cpu().numpy()
    bin_mask = (mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(bin_mask)
    mask_img = mask_img.resize(orig_size, Image.NEAREST)
    mask_img.save(path)

def main():
    # 1. Модель
    model = SE_PP_UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, base_channels=BASE_CHANNELS)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt.get('state_dict', ckpt)
    clean = OrderedDict()
    for k, v in state.items():
        clean[k.replace('module.', '')] = v
    model.load_state_dict(clean, strict=False)
    model.to(DEVICE)
    model.eval()

    # 2. Поиск всех картинок
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG'):
        image_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))

    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, f'{name}_mask.png')

        # --- Генерация и сохранение маски с помощью модели всегда ---
        image_tensor, orig_size = load_image(img_path)
        image_tensor = image_tensor.to(DEVICE)
        with torch.no_grad():
            pred = model(image_tensor)
            pred = torch.sigmoid(pred)
        save_mask(pred, orig_size, mask_path)
        print(f"Mask generated and saved: {mask_path}")

        # Read original и mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Threshold mask (in case not binary)
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours over the image
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

        # Save
        out_path = os.path.join(OUT_DIR, f'{name}_contour.png')
        cv2.imwrite(out_path, overlay)
        print(f'Saved: {out_path}')

if __name__ == '__main__':
    main()
