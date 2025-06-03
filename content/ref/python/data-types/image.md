---
title: Image
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.0/wandb/sdk/data_types/image.py#L117-L769 >}}

A class for logging images to W&B.

```python
Image(
    data_or_path: "ImageDataOrPathType",
    mode: Optional[str] = None,
    caption: Optional[str] = None,
    grouping: Optional[int] = None,
    classes: Optional[Union['Classes', Sequence[dict]]] = None,
    boxes: Optional[Union[Dict[str, 'BoundingBoxes2D'], Dict[str, dict]]] = None,
    masks: Optional[Union[Dict[str, 'ImageMask'], Dict[str, dict]]] = None,
    file_type: Optional[str] = None,
    normalize: bool = (True)
) -> None
```

| Args |  |
| :--- | :--- |
|  `data_or_path` |  Accepts numpy array/pytorch tensor of image data, a PIL image object, or a path to an image file. If a numpy array or pytorch tensor is provided, the image data will be saved to the given file type. If the values are not in the range [0, 255] or all values are in the range [0, 1], the image pixel values will be normalized to the range [0, 255] unless `normalize` is set to False. - pytorch tensor should be in the format (channel, height, width) - numpy array should be in the format (height, width, channel) |
|  `mode` |  The PIL mode for an image. Most common are "L", "RGB", "RGBA". Full explanation at https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes |
|  `caption` |  Label for display of image. |
|  `grouping` |  The grouping number for the image. |
|  `classes` |  A list of class information for the image, used for labeling bounding boxes, and image masks. |
|  `boxes` |  A dictionary containing bounding box information for the image. see: https://docs.wandb.ai/ref/python/data-types/boundingboxes2d/ |
|  `masks` |  A dictionary containing mask information for the image. see: https://docs.wandb.ai/ref/python/data-types/imagemask/ |
|  `file_type` |  The file type to save the image as. This parameter has no effect if data_or_path is a path to an image file. |
|  `normalize` |  If True, normalize the image pixel values to fall within the range of [0, 255]. Normalize is only applied if data_or_path is a numpy array or pytorch tensor. |

| Attributes |  |
| :--- | :--- |

## Methods

### `all_boxes`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.0/wandb/sdk/data_types/image.py#L690-L711)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.20.0/wandb/sdk/data_types/image.py#L713-L717)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

### `all_masks`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.0/wandb/sdk/data_types/image.py#L667-L688)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.20.0/wandb/sdk/data_types/image.py#L560-L591)

```python
guess_mode(
    data: Union['np.ndarray', 'torch.Tensor'],
    file_type: Optional[str] = None
) -> str
```

Guess what type of image the np.array is representing.

| Class Variables |  |
| :--- | :--- |
|  `MAX_DIMENSION`<a id="MAX_DIMENSION"></a> |  `65500` |
|  `MAX_ITEMS`<a id="MAX_ITEMS"></a> |  `108` |
