"""
Checkpoint and Model Export Utilities

Provides comprehensive model saving/loading with multiple format support:
- PyTorch (.pt, .pth) - Native format
- TorchScript (.ts) - For production deployment
- ONNX (.onnx) - Cross-platform inference
- SafeTensors (.safetensors) - Safe, fast loading
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import torch
import torch.nn as nn

# Try importing optional dependencies
try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from safetensors.torch import save_file as save_safetensors
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ==============================================================================
# Basic Checkpoint Saving/Loading
# ==============================================================================

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    folder: str = 'checkpoints',
    filename: str = 'checkpoint.pth'
) -> str:
    """
    Save training checkpoint with optional best model copy.
    
    Args:
        state: Dictionary containing model state_dict, optimizer, epoch, etc.
        is_best: If True, also save as best_model.pth
        folder: Directory to save checkpoints
        filename: Name of checkpoint file
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(folder, 'best_model.pth')
        shutil.copyfile(filepath, best_path)
    
    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        device: Device to load model to
        
    Returns:
        Checkpoint dictionary with epoch, loss, etc.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint


# ==============================================================================
# Multiple Format Export
# ==============================================================================

def save_model(
    model: nn.Module,
    filepath: str,
    format: str = 'pt',
    input_shape: tuple = (1, 3, 256, 256),
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save model in specified format.
    
    Args:
        model: PyTorch model to save
        filepath: Output path (extension will be adjusted based on format)
        format: One of 'pt', 'pth', 'torchscript', 'onnx', 'safetensors'
        input_shape: Input tensor shape for ONNX/TorchScript export
        metadata: Optional metadata to save alongside model
        
    Returns:
        Path to saved model
    """
    model.eval()
    filepath = Path(filepath)
    base_path = filepath.parent / filepath.stem
    
    if format in ('pt', 'pth'):
        return _save_pytorch(model, f"{base_path}.{format}", metadata)
    elif format == 'torchscript':
        return _save_torchscript(model, f"{base_path}.ts", input_shape)
    elif format == 'onnx':
        return _save_onnx(model, f"{base_path}.onnx", input_shape, metadata)
    elif format == 'safetensors':
        return _save_safetensors_format(model, f"{base_path}.safetensors", metadata)
    else:
        raise ValueError(f"Unknown format: {format}. Supported: pt, pth, torchscript, onnx, safetensors")


def _save_pytorch(
    model: nn.Module,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save as PyTorch format (.pt/.pth)."""
    save_dict = {
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, filepath)
    print(f"✅ Saved PyTorch model: {filepath}")
    return filepath


def _save_torchscript(
    model: nn.Module,
    filepath: str,
    input_shape: tuple
) -> str:
    """Save as TorchScript format (.ts)."""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    
    try:
        # Try tracing first (faster, works for most models)
        traced = torch.jit.trace(model, dummy_input)
    except Exception:
        # Fall back to scripting (handles dynamic control flow)
        traced = torch.jit.script(model)
    
    traced.save(filepath)
    print(f"✅ Saved TorchScript model: {filepath}")
    return filepath


def _save_onnx(
    model: nn.Module,
    filepath: str,
    input_shape: tuple,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save as ONNX format (.onnx)."""
    if not HAS_ONNX:
        raise ImportError("ONNX not installed. Run: pip install onnx onnxruntime")
    
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    # Verify the model
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ Saved ONNX model: {filepath}")
    
    # Save metadata alongside
    if metadata:
        meta_path = filepath.replace('.onnx', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return filepath


def _save_safetensors_format(
    model: nn.Module,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save as SafeTensors format (.safetensors)."""
    if not HAS_SAFETENSORS:
        raise ImportError("SafeTensors not installed. Run: pip install safetensors")
    
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    
    # SafeTensors metadata must be strings
    meta = {}
    if metadata:
        for k, v in metadata.items():
            meta[k] = str(v)
    meta['model_class'] = model.__class__.__name__
    meta['timestamp'] = datetime.now().isoformat()
    
    save_safetensors(state_dict, filepath, metadata=meta)
    print(f"✅ Saved SafeTensors model: {filepath}")
    return filepath


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(
    filepath: str,
    model: nn.Module,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load model from various formats.
    
    Args:
        filepath: Path to model file
        model: Model instance to load weights into
        device: Device to load to
        
    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    filepath = str(filepath)
    
    if filepath.endswith('.safetensors'):
        if not HAS_SAFETENSORS:
            raise ImportError("SafeTensors not installed")
        state_dict = load_safetensors(filepath, device=str(device))
        model.load_state_dict(state_dict)
        
    elif filepath.endswith('.ts'):
        # For TorchScript, return the loaded model directly
        return torch.jit.load(filepath, map_location=device)
        
    else:  # .pt, .pth
        checkpoint = torch.load(filepath, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


# ==============================================================================
# Multi-format Export
# ==============================================================================

def export_all_formats(
    model: nn.Module,
    output_dir: str,
    model_name: str,
    input_shape: tuple = (1, 3, 256, 256),
    metadata: Optional[Dict[str, Any]] = None,
    formats: tuple = ('pt', 'torchscript', 'onnx')
) -> Dict[str, str]:
    """
    Export model to multiple formats at once.
    
    Args:
        model: Model to export
        output_dir: Output directory
        model_name: Base name for exported files
        input_shape: Input tensor shape
        metadata: Optional metadata
        formats: Tuple of formats to export
        
    Returns:
        Dictionary mapping format to filepath
    """
    os.makedirs(output_dir, exist_ok=True)
    exported = {}
    
    for fmt in formats:
        try:
            filepath = os.path.join(output_dir, model_name)
            path = save_model(model, filepath, format=fmt, input_shape=input_shape, metadata=metadata)
            exported[fmt] = path
        except Exception as e:
            print(f"⚠️  Failed to export {fmt}: {e}")
    
    print(f"\n📦 Exported {len(exported)} formats to {output_dir}")
    return exported


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter count and size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Size in MB (assuming float32)
    size_mb = total_params * 4 / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'size_mb': round(size_mb, 2)
    }


def create_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    **kwargs
) -> Dict[str, Any]:
    """
    Create a comprehensive training checkpoint.
    
    Args:
        model: Trained model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Dictionary of metrics
        **kwargs: Additional items to save
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'model_class': model.__class__.__name__,
        'model_info': get_model_size(model),
    }
    checkpoint.update(kwargs)
    return checkpoint


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    # Create a simple test model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 3, padding=1)
    )
    
    print("Model Info:", get_model_size(model))
    
    # Export to multiple formats
    export_all_formats(
        model,
        output_dir='./test_exports',
        model_name='test_model',
        input_shape=(1, 3, 64, 64),
        metadata={'version': '1.0', 'task': 'segmentation'}
    )
