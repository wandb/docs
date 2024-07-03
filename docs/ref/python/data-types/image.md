# Image

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/image.py#L64-L687' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&Bへのログに画像をフォーマットします。

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

| 引数 |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) 画像データのnumpy配列、またはPIL画像を受け入れます。クラスはデータ形式を推測し、変換を行います。 |
|  `mode` |  (string) 画像のPILモード。最も一般的なのは "L", "RGB", "RGBA" です。詳細は https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes をご覧ください。 |
|  `caption` |  (string) 画像表示のためのラベル。 |

注意 : `torch.Tensor` を `wandb.Image` としてログに記録する際、画像は正規化されます。画像を正規化したくない場合は、テンソルをPIL画像に変換してください。

#### 例:

### numpy arrayからwandb.Imageを作成する

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

### PILImageからwandb.Imageを作成する

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

### .png（デフォルト）ではなく .jpg をログに記録する

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

| 属性 |  |
| :--- | :--- |

## メソッド

### `all_boxes`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/image.py#L608-L629)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/image.py#L631-L635)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

### `all_masks`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/image.py#L585-L606)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/image.py#L472-L484)

```python
guess_mode(
    data: "np.ndarray"
) -> str
```

np.array がどのタイプの画像を表しているか推測します。

### `to_uint8`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/image.py#L486-L509)

```python
@classmethod
to_uint8(
    data: "np.ndarray"
) -> "np.ndarray"
```

画像データをuint8に変換します。

範囲 [0,1] の浮動小数点画像と範囲 [0,255] の整数画像を uint8 に変換し、必要に応じてクリッピングします。

| クラス変数 |  |
| :--- | :--- |
|  `MAX_DIMENSION`<a id="MAX_DIMENSION"></a> |  `65500` |
|  `MAX_ITEMS`<a id="MAX_ITEMS"></a> |  `108` |