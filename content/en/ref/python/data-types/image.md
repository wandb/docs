---
title: Image
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/image.py#L131-L783 >}}

A class for logging images to W&B.

| Attributes |  |
| :--- | :--- |

## Methods

### `all_boxes`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/image.py#L704-L725)

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

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/image.py#L727-L731)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

### `all_masks`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/image.py#L681-L702)

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

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/image.py#L574-L605)

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
