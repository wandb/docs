---
title: "Image Normalization Guide"
description: "Learn how wandb.Image handles normalization for different input types and how to control this behavior"
---

# Image Normalization Guide

When you pass PyTorch tensors or NumPy arrays to `wandb.Image`, the pixel values are automatically normalized to the range [0, 255] unless you set `normalize=False`. This guide explains how normalization works and how to control it.

## When normalization is applied

Normalization is applied to:
- **PyTorch tensors** (format: `(channel, height, width)`)
- **NumPy arrays** (format: `(height, width, channel)`)

Normalization is **NOT** applied to:
- **PIL Images** (passed as-is)
- **File paths** (loaded as-is)

## Normalization algorithm

The normalization algorithm automatically detects the input range and applies the appropriate transformation:

1. **If data is in range [0, 1]**: Values are multiplied by 255 and converted to uint8
   ```python
   normalized_data = (data * 255).astype(np.uint8)
   ```

2. **If data is in range [-1, 1]**: Values are rescaled to [0, 255] using:
   ```python
   normalized_data = (255 * 0.5 * (data + 1)).astype(np.uint8)
   ```

3. **For any other range**: Values are clipped to [0, 255] and converted to uint8
   ```python
   normalized_data = data.clip(0, 255).astype(np.uint8)
   ```

## Examples of normalization effects

### Example 1: [0, 1] range data

```python
import torch
import wandb

# Create tensor with values in [0, 1] range
tensor_0_1 = torch.rand(3, 64, 64)  # Random values between 0 and 1

# This will multiply all values by 255
image = wandb.Image(tensor_0_1, caption="Normalized from [0,1] range")
```

### Example 2: [-1, 1] range data

```python
import torch
import wandb

# Create tensor with values in [-1, 1] range
tensor_neg1_1 = torch.rand(3, 64, 64) * 2 - 1  # Random values between -1 and 1

# This will rescale: -1 → 0, 0 → 127.5, 1 → 255
image = wandb.Image(tensor_neg1_1, caption="Normalized from [-1,1] range")
```

### Example 3: Avoiding normalization with PIL Images

```python
import torch
from PIL import Image as PILImage
import wandb

# Create tensor with values in [0, 1] range
tensor_0_1 = torch.rand(3, 64, 64)

# Convert to PIL Image to avoid normalization
pil_image = PILImage.fromarray((tensor_0_1.permute(1, 2, 0).numpy() * 255).astype('uint8'))
image = wandb.Image(pil_image, caption="No normalization applied")
```

### Example 4: Using normalize=False

```python
import torch
import wandb

# Create tensor with values in [0, 1] range
tensor_0_1 = torch.rand(3, 64, 64)

# Disable normalization - values will be clipped to [0, 255]
image = wandb.Image(tensor_0_1, normalize=False, caption="Normalization disabled")
```

## Best practices

1. **For consistent results**: Pre-process your data to the expected [0, 255] range before logging
2. **To avoid normalization**: Convert tensors to PIL Images using `PILImage.fromarray()`
3. **For debugging**: Use `normalize=False` to see the raw values (they will be clipped to [0, 255])
4. **For precise control**: Use PIL Images when you need exact pixel values

## Common issues

- **Unexpected brightness**: If your tensor values are in [0, 1] range, they will be multiplied by 255, making the image much brighter
- **Data loss**: Values outside the [0, 255] range will be clipped, potentially losing information
- **Inconsistent behavior**: Different input types (tensor vs PIL vs file path) may produce different results

## Testing your code

You can test the normalization behavior using our [Image Normalization Demo Notebook](https://github.com/wandb/wandb/blob/main/wandb_image_normalization_demo.ipynb) which demonstrates all the examples above with visual output. 