# Image



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/v0.15.5/wandb/sdk/data_types/image.py#L64-L657)



Format images for logging to W&B.

```python
Image(
 data_or_path: "ImageDataOrPathType",
 mode: Optional[str] = None,
 caption: Optional[str] = None,
 grouping: Optional[int] = None,
 classes: Optional[Union['Classes', Sequence[dict]]] = None,
 boxes: Optional[Union[Dict[str, 'BoundingBoxes2D'], Dict[str, dict]]] = None,
 masks: Optional[Union[Dict[str, 'ImageMask'], Dict[str, dict]]] = None
) -> None
```





| Arguments | |
| :--- | :--- |
| `data_or_path` | (numpy array, string, io) Accepts numpy array of image data, or a PIL image. The class attempts to infer the data format and converts it. |
| `mode` | (string) The PIL mode for an image. Most common are "L", "RGB", "RGBA". Full explanation at https://pillow.readthedocs.io/en/4.2.x/handbook/concepts.html#concept-modes. |
| `caption` | (string) Label for display of image. |


Note : When logging a `torch.Tensor` as a `wandb.Image`, images are normalized. If you do not want to normalize your images, please convert your tensors to a PIL Image.

#### Examples:

### Create a wandb.Image from a numpy array

```python
import numpy as np
import wandb

wandb.init()
examples = []
for i in range(3):
 pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
 image = wandb.Image(pixels, caption=f"random field {i}")
 examples.append(image)
wandb.log({"examples": examples})
```

### Create a wandb.Image from a PILImage

```python
import numpy as np
from PIL import Image as PILImage
import wandb

wandb.init()
examples = []
for i in range(3):
 pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
 pil_image = PILImage.fromarray(pixels, mode="RGB")
 image = wandb.Image(pil_image, caption=f"random field {i}")
 examples.append(image)
wandb.log({"examples": examples})
```




| Attributes | |
| :--- | :--- |



## Methods

### `all_boxes`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/sdk/data_types/image.py#L578-L599)

```python
@classmethod
all_boxes(
 images: Sequence['Image'],
 run: "LocalRun",
 run_key: str,
 step: Union[int, str]
) -> Union[List[Optional[dict]], bool]
```




### `all_captions`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/sdk/data_types/image.py#L601-L605)

```python
@classmethod
all_captions(
 images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```




### `all_masks`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/sdk/data_types/image.py#L555-L576)

```python
@classmethod
all_masks(
 images: Sequence['Image'],
 run: "LocalRun",
 run_key: str,
 step: Union[int, str]
) -> Union[List[Optional[dict]], bool]
```




### `guess_mode`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/sdk/data_types/image.py#L442-L454)

```python
guess_mode(
 data: "np.ndarray"
) -> str
```

Guess what type of image the np.array is representing.


### `to_uint8`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/sdk/data_types/image.py#L456-L479)

```python
@classmethod
to_uint8(
 data: "np.ndarray"
) -> "np.ndarray"
```

Convert image data to uint8.

Convert floating point image on the range [0,1] and integer images on the range
[0,255] to uint8, clipping if necessary.





| Class Variables | |
| :--- | :--- |
| `MAX_DIMENSION` | `65500` |
| `MAX_ITEMS` | `108` |

