---
title: Image
menu:
  reference:
    identifier: ja-ref-python-data-types-image
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L65-L689 >}}

W&B に ログ記録 するための画像形式。

```python
Image(
    data_or_path: "ImageDataOrPathType",
    mode: Optional[str] = None,
    caption: Optional[str] = None,
    grouping: Optional[int] = None,
    classes: Optional[Union['Classes', Sequence[dict]]] = None,
    boxes: Optional[Union[Dict[str, 'BoundingBoxes2D'], Dict[str, dict]]] = None,
    masks: Optional[Union[Dict[str, 'ImageMask'], Dict[str, dict]]] = None,
    file_type: Optional[str] = None
) -> None
```

| arg |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) 画像データの numpy array、または PIL image を受け入れます。クラスはデータ形式の推測を試み、変換します。 |
|  `mode` |  (string) 画像の PIL モード。最も一般的なものは "L"、"RGB"、"RGBA" です。詳細な説明は https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes を参照してください。 |
|  `caption` |  (string) 画像表示用のラベル。 |

Note : `torch.Tensor` を `wandb.Image` として ログ記録 する場合、画像は正規化されます。画像を正規化しない場合は、テンソルを PIL Image に変換してください。

#### Examples:

### numpy array から wandb.Image を作成する

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

### PILImage から wandb.Image を作成する

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

### .png (デフォルト) ではなく .jpg を ログ記録 する

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

| Attributes |  |
| :--- | :--- |

## メソッド

### `all_boxes`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L610-L631)

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

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L633-L637)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

### `all_masks`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L587-L608)

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

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L474-L486)

```python
guess_mode(
    data: "np.ndarray"
) -> str
```

np.array が表す画像のタイプを推測します。

### `to_uint8`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L488-L511)

```python
@classmethod
to_uint8(
    data: "np.ndarray"
) -> "np.ndarray"
```

画像データを uint8 に変換します。

[0,1] の範囲の浮動小数点画像と、[0,255] の範囲の整数画像を uint8 に変換し、必要に応じてクリップします。

| Class Variables |  |
| :--- | :--- |
|  `MAX_DIMENSION`<a id="MAX_DIMENSION"></a> |  `65500` |
|  `MAX_ITEMS`<a id="MAX_ITEMS"></a> |  `108` |
