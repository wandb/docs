---
title: Image
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/image.py >}}




## <kbd>class</kbd> `Image`
A class for logging images to W&B. 

### <kbd>method</kbd> `Image.__init__`

```python
__init__(
    data_or_path: 'ImageDataOrPathType',
    mode: Optional[str] = None,
    caption: Optional[str] = None,
    grouping: Optional[int] = None,
    classes: Optional[ForwardRef('Classes'), Sequence[dict]] = None,
    boxes: Optional[Dict[str, ForwardRef('BoundingBoxes2D')], Dict[str, dict]] = None,
    masks: Optional[Dict[str, ForwardRef('ImageMask')], Dict[str, dict]] = None,
    file_type: Optional[str] = None,
    normalize: bool = True
) → None
```

Initialize a `wandb.Image` object. 



**Args:**
 
 - `data_or_path`:  Accepts NumPy array/pytorch tensor of image data,  a PIL image object, or a path to an image file. If a NumPy  array or pytorch tensor is provided,  the image data will be saved to the given file type.  If the values are not in the range [0, 255] or all values are in the range [0, 1],  the image pixel values will be normalized to the range [0, 255]  unless `normalize` is set to False. 
    - pytorch tensor should be in the format (channel, height, width) 
    - NumPy array should be in the format (height, width, channel) 
 - `mode`:  The PIL mode for an image. Most common are "L", "RGB", 
 - `"RGBA". Full explanation at https`: //pillow.readthedocs.io/en/stable/handbook/concepts.html#modes 
 - `caption`:  Label for display of image. 
 - `grouping`:  The grouping number for the image. 
 - `classes`:  A list of class information for the image,  used for labeling bounding boxes, and image masks. 
 - `boxes`:  A dictionary containing bounding box information for the image. 
 - `see`:  https://docs.wandb.ai/ref/python/data-types/boundingboxes2d/ 
 - `masks`:  A dictionary containing mask information for the image. 
 - `see`:  https://docs.wandb.ai/ref/python/data-types/imagemask/ 
 - `file_type`:  The file type to save the image as.  This parameter has no effect if data_or_path is a path to an image file. 
 - `normalize`:  If True, normalize the image pixel values to fall within the range of [0, 255].  Normalize is only applied if data_or_path is a numpy array or pytorch tensor. 



**Examples:**
 Create a wandb.Image from a numpy array 

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(pixels, caption=f"random field {i}")
         examples.append(image)
    run.log({"examples": examples})
``` 

Create a wandb.Image from a PILImage 

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         pixels = np.random.randint(
             low=0, high=256, size=(100, 100, 3), dtype=np.uint8
         )
         pil_image = PILImage.fromarray(pixels, mode="RGB")
         image = wandb.Image(pil_image, caption=f"random field {i}")
         examples.append(image)
    run.log({"examples": examples})
``` 

Log .jpg rather than .png (default) 

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(
             pixels, caption=f"random field {i}", file_type="jpg"
         )
         examples.append(image)
    run.log({"examples": examples})
``` 

## Image normalization

When you pass PyTorch tensors or NumPy arrays to `wandb.Image`, the pixel values are automatically normalized to the range [0, 255] unless you set `normalize=False`. This normalization is designed to handle and ensure proper display of common image formats.

### When normalization is applied

Normalization is applied to:
- **PyTorch tensors** (format: `(channel, height, width)`)
- **NumPy arrays** (format: `(height, width, channel)`)

Normalization is **NOT** applied to:
- **PIL Images** (passed as-is)
- **File paths** (loaded as-is)

### Normalization algorithm

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

### Examples of normalization effects

**Example 1: [0, 1] range data**
```python
import torch
import wandb

# Create tensor with values in [0, 1] range
tensor_0_1 = torch.rand(3, 64, 64)  # Random values between 0 and 1

# This will multiply all values by 255
image = wandb.Image(tensor_0_1, caption="Normalized from [0,1] range")
```

**Example 2: [-1, 1] range data**
```python
import torch
import wandb

# Create tensor with values in [-1, 1] range
tensor_neg1_1 = torch.rand(3, 64, 64) * 2 - 1  # Random values between -1 and 1

# This will rescale: -1 → 0, 0 → 127.5, 1 → 255
image = wandb.Image(tensor_neg1_1, caption="Normalized from [-1,1] range")
```

**Example 3: Avoiding normalization with PIL Images**
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

**Example 4: Using normalize=False**
```python
import torch
import wandb

# Create tensor with values in [0, 1] range
tensor_0_1 = torch.rand(3, 64, 64)

# Disable normalization - values will be clipped to [0, 255]
image = wandb.Image(tensor_0_1, normalize=False, caption="Normalization disabled")
```

### Recommendations

1. **For consistent results**: Pre-process your data to the expected [0, 255] range before logging
2. **To avoid normalization**: Convert tensors to PIL Images using `PILImage.fromarray()`
3. **For debugging**: Use `normalize=False` to see the raw values (they will be clipped to [0, 255])
4. **For precise control**: Use PIL Images when you need exact pixel values

### Troubleshooting

- **Unexpected brightness**: If your tensor values are in [0, 1] range, they will be multiplied by 255, making the image much brighter
- **Data loss**: Values outside the [0, 255] range will be clipped, potentially losing information
- **Inconsistent behavior**: Different input types (tensor vs PIL vs file path) may produce different results

---

### <kbd>property</kbd> Image.image







---






