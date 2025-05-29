---
title: Image
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/image.py >}}




## <kbd>class</kbd> `Image`
A class for logging images to W&B.. 

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
    file_type: Optional[str] = None,
    normalize: bool = True
) → None
```

Initialize a wandb.Image object. 



**Args:**
 
 - `data_or_path`:  Accepts numpy array/pytorch tensor of image data,  a PIL image object, or a path to an image file. 

 If a numpy array or pytorch tensor is provided,  the image data will be saved to the given file type.  If the values are not in the range [0, 255] or all values are in the range [0, 1],  the image pixel values will be normalized to the range [0, 255]  unless `normalize` is set to False. 
        - pytorch tensor should be in the format (channel, height, width) 
        - numpy array should be in the format (height, width, channel) 
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
 ### Create a wandb.Image from a numpy array ```python
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

### Create a wandb.Image from a PILImage ```python
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

### log .jpg rather than .png (default) ```python
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



