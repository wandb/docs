# 画像

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/image.py#L64-L655)

W&Bにログを送るための画像を整形します。

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

| 引数 |  |
| :--- | :--- |
| `data_or_path` | (numpy配列, 文字列, io) 画像データのnumpy配列またはPILイメージを受け取ります。クラスはデータフォーマットを推測し、それに変換します。 |
| `mode` | (文字列) 画像のPILモード。最も一般的なものは "L"、"RGB"、"RGBA"です。詳細はhttps://pillow.readthedocs.io/en/4.2.x/handbook/concepts.html#concept-modes で確認できます。|
| `caption` | (文字列) 画像の表示用ラベル。|
注意: `torch.Tensor` を `wandb.Image` としてログに記録する際、画像は正規化されます。画像を正規化したくない場合は、テンソルをPILイメージに変換してください。

#### 例:

### numpy配列からwandb.Imageを作成する

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

### PILImageからwandb.Imageを作成する

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
| 属性 | |
| :--- | :--- |

## メソッド

### `all_boxes`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/image.py#L576-L597)

```python
@classmethod
all_boxes(
 画像: Sequence['Image'],
 run: "LocalRun",
 run_key: str,
 ステップ: Union[int, str]
) -> Union[List[Optional[dict]], bool]
```
### `all_captions`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/image.py#L599-L603)

```python
@classmethod
all_captions(
 images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```




### `all_masks`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/image.py#L553-L574)

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

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/image.py#L440-L452)

```python
guess_mode(
 data: "np.ndarray"
) -> str
```

np.arrayで表現される画像の種類を推測します。

### `to_uint8`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/image.py#L454-L477)

```python
@classmethod
to_uint8(
 data: "np.ndarray"
) -> "np.ndarray"
```
画像データをuint8に変換します。



範囲[0,1]の浮動小数点画像および範囲

[0,255]の整数画像をuint8に変換し、必要に応じてクリッピングします。





| クラス変数 | |

| :--- | :--- |

| `MAX_DIMENSION` | `65500` |

| `MAX_ITEMS` | `108` |