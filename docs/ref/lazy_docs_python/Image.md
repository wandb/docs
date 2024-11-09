import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Image

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/image.py'/>




## <kbd>class</kbd> `Image`
Format images for logging to W&B. 



**Arguments:**
 
 - `data_or_path`:  (numpy array, string, io) Accepts numpy array of  image data, or a PIL image. The class attempts to infer  the data format and converts it. 
 - `mode`:  (string) The PIL mode for an image. Most common are "L", "RGB", 
 - `"RGBA". Full explanation at https`: //pillow.readthedocs.io/en/stable/handbook/concepts.html#modes 
 - `caption`:  (string) Label for display of image. 

Note : When logging a `torch.Tensor` as a `wandb.Image`, images are normalized. If you do not want to normalize your images, please convert your tensors to a PIL Image. 



**Examples:**

### Create a wandb.Image from a numpy array 
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

### Create a wandb.Image from a PILImage
```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
        examples = []
        for i in range(3):
            pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
            pil_image = PILImage.fromarray(pixels, mode="RGB")
            image = wandb.Image(pil_image, caption=f"random field {i}")
            examples.append(image)
        run.log({"examples": examples})
``` 

### log .jpg rather than .png (default) 
```python
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

#### <kbd>property</kbd> Image.image







---

### <kbd>classmethod</kbd> `Image.all_boxes`

```python
all_boxes(
    images: Sequence[ForwardRef('Image')],
    run: 'LocalRun',
    run_key: str,
    step: Union[int, str]
) → Union[List[Optional[dict]], bool]
```





---

### <kbd>classmethod</kbd> `Image.all_captions`

```python
all_captions(
    images: Sequence[ForwardRef('Media')]
) → Union[bool, Sequence[Optional[str]]]
```





---

### <kbd>classmethod</kbd> `Image.all_masks`

```python
all_masks(
    images: Sequence[ForwardRef('Image')],
    run: 'LocalRun',
    run_key: str,
    step: Union[int, str]
) → Union[List[Optional[dict]], bool]
```





---

### <kbd>method</kbd> `Image.bind_to_run`

```python
bind_to_run(
    run: 'LocalRun',
    key: Union[int, str],
    step: Union[int, str],
    id_: Optional[int, str] = None,
    ignore_copy_err: Optional[bool] = None
) → None
```





---

### <kbd>classmethod</kbd> `Image.from_json`

```python
from_json(json_obj: dict, source_artifact: 'Artifact') → Image
```





---

### <kbd>classmethod</kbd> `Image.get_media_subdir`

```python
get_media_subdir() → str
```





---

### <kbd>method</kbd> `Image.guess_mode`

```python
guess_mode(data: 'np.ndarray') → str
```

Guess what type of image the np.array is representing. 

---

### <kbd>classmethod</kbd> `Image.seq_to_json`

```python
seq_to_json(
    seq: Sequence[ForwardRef('BatchableMedia')],
    run: 'LocalRun',
    key: str,
    step: Union[int, str]
) → dict
```

Combine a list of images into a meta dictionary object describing the child images. 

---

### <kbd>method</kbd> `Image.to_data_array`

```python
to_data_array() → List[Any]
```





---

### <kbd>method</kbd> `Image.to_json`

```python
to_json(
    run_or_artifact: Union[ForwardRef('LocalRun'), ForwardRef('Artifact')]
) → dict
```





---

### <kbd>classmethod</kbd> `Image.to_uint8`

```python
to_uint8(data: 'np.ndarray') → np.ndarray
```

Convert image data to uint8. 

Convert floating point image on the range [0,1] and integer images on the range [0,255] to uint8, clipping if necessary.