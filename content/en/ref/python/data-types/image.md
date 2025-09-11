---
title: Image
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/image.py#L129-L819 >}}

A class for logging images to W&B.

| Attributes |  |
| :--- | :--- |

## Methods

### `all_boxes`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/image.py#L728-L753)

```python
@classmethod
all_boxes(
    images: Sequence['Image'],
    run: "wandb.Run",
    run_key: str,
    step: Union[int, str]
) -> Union[List[Optional[dict]], bool]
```

Collect all boxes from a list of images.

"<!-- lazydoc-ignore-classmethod: internal -->

### `all_captions`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/image.py#L755-L763)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

Get captions from a list of images.

"<!-- lazydoc-ignore-classmethod: internal -->

### `all_masks`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/image.py#L701-L726)

```python
@classmethod
all_masks(
    images: Sequence['Image'],
    run: "wandb.Run",
    run_key: str,
    step: Union[int, str]
) -> Union[List[Optional[dict]], bool]
```

Collect all masks from a list of images.

"<!-- lazydoc-ignore-classmethod: internal -->

### `guess_mode`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/image.py#L588-L622)

```python
guess_mode(
    data: Union['np.ndarray', 'torch.Tensor'],
    file_type: Optional[str] = None
) -> str
```

Guess what type of image the np.array is representing.

<!-- lazydoc-ignore: internal -->


| Class Variables |  |
| :--- | :--- |
|  `MAX_DIMENSION`<a id="MAX_DIMENSION"></a> |  `65500` |
|  `MAX_ITEMS`<a id="MAX_ITEMS"></a> |  `108` |
