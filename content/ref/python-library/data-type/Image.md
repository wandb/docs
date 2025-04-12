---
title: Image
object_type: data-type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/image.py >}}




## <kbd>class</kbd> `Image`
Format images for logging to W&B. 

See https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes for more information on modes. 



**Args:**
 
 - `data_or_path`:  Accepts numpy array of image data, or a PIL image.  The class attempts to infer the data format and converts it. 
 - `mode`:  The PIL mode for an image. Most common are "L", "RGB", "RGBA". 
 - `caption`:  Label for display of image. 

When logging a `torch.Tensor` as a `wandb.Image`, images are normalized. If you do not want to normalize your images, convert your tensors to a PIL Image. 



**Examples:**
 ```python
# Create a wandb.Image from a numpy array
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

```python
# Create a wandb.Image from a PILImage
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

```python
# log .jpg rather than .png (default)
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(pixels, caption=f"random field {i}", file_type="jpg")
         examples.append(image)
    run.log({"examples": examples})
``` 

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
    file_type: Optional[str] = None
) → None
```






---







### <kbd>method</kbd> `Image.guess_mode`

```python
guess_mode(
    data: Union[ForwardRef('np.ndarray'), ForwardRef('torch.Tensor')],
    file_type: Optional[str] = None
) → str
```

Guess what type of image the np.array is representing. 

---




