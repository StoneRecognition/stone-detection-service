---
description: How to add a new model to this project
---

# Adding a New Model

## Steps

1. **Create model file** in `src/models/`
   ```bash
   # Example: src/models/my_new_model.py
   ```

2. **Model template**:
   ```python
   import torch
   import torch.nn as nn
   from typing import Tuple

   class MyNewModel(nn.Module):
       """
       Brief description of the model.
       
       Args:
           in_channels: Number of input channels
           out_channels: Number of output classes
       """
       
       def __init__(self, in_channels: int = 3, out_channels: int = 1):
           super().__init__()
           # Define layers
           
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """
           Forward pass.
           
           Args:
               x: Input tensor of shape (B, C, H, W)
               
           Returns:
               Output tensor of shape (B, out_channels, H, W)
           """
           return x
   ```

3. **Update package exports** in `src/models/__init__.py`:
   ```python
   from .my_new_model import MyNewModel
   ```

4. **Create training script** (if needed) in `src/training/`:
   ```bash
   # Example: src/training/train_my_model.py
   ```

5. **Add requirements** (if new dependencies) to `requirements/`:
   ```bash
   # Example: requirements/requirements.my_model.txt
   ```

6. **Create inference script** in `src/inference/`:
   ```bash
   # Example: src/inference/my_model_inference.py
   ```

7. **Test the model**:
   ```python
   # In tests/test_my_model.py
   def test_model_forward():
       model = MyNewModel()
       x = torch.randn(1, 3, 256, 256)
       out = model(x)
       assert out.shape == (1, 1, 256, 256)
   ```
