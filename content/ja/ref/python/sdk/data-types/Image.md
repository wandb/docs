---
title: 画像
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/image.py >}}




## <kbd>class</kbd> `Image`
W&B に画像をログするためのクラスです。

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

`wandb.Image` オブジェクトを初期化します。



**パラメータ:**
 
 - `data_or_path`:  NumPy 配列/pytorch テンソルの画像データ、PIL イメージオブジェクト、または画像ファイルへのパスを指定できます。NumPy 配列または pytorch テンソルが指定された場合、画像データは指定されたファイルタイプで保存されます。値が [0, 255] の範囲外、または全ての値が [0, 1] の範囲内であれば、`normalize` が False でない限りピクセル値は [0, 255] の範囲に正規化されます。
    - pytorch テンソルの場合のフォーマットは (channel, height, width) です。
    - NumPy 配列の場合は (height, width, channel) です。
 - `mode`:  PIL 画像のモード。最も一般的なのは "L"、"RGB"、"RGBA" です。詳細は https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes を参照してください。
 - `caption`:  画像を表示する際のラベルです。
 - `grouping`:  画像のグルーピング番号です。
 - `classes`:  画像のクラス情報のリスト。バウンディングボックスや画像マスクのラベリングに使用します。
 - `boxes`:  画像のバウンディングボックス情報を含む辞書です。
 - `see`:  https://docs.wandb.ai/ref/python/data-types/boundingboxes2d/
 - `masks`:  画像のマスク情報を含む辞書です。
 - `see`:  https://docs.wandb.ai/ref/python/data-types/imagemask/
 - `file_type`:  画像を保存するファイルタイプ。このパラメータは、data_or_path が画像ファイルのパスの場合は影響しません。
 - `normalize`:  True の場合、画像ピクセル値を [0, 255] の範囲に正規化します。`normalize` は data_or_path が numpy 配列または pytorch テンソルの場合のみ適用されます。



**例:**
 numpy 配列から wandb.Image を作成

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         # ランダムな画像データを生成
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(pixels, caption=f"random field {i}")
         examples.append(image)
    run.log({"examples": examples})
```

PILImage から wandb.Image を作成

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         # ランダムな画像データを生成
         pixels = np.random.randint(
             low=0, high=256, size=(100, 100, 3), dtype=np.uint8
         )
         pil_image = PILImage.fromarray(pixels, mode="RGB")
         image = wandb.Image(pil_image, caption=f"random field {i}")
         examples.append(image)
    run.log({"examples": examples})
```

デフォルト（.png）ではなく .jpg でログする

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         # ランダムな画像データを生成
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(
             pixels, caption=f"random field {i}", file_type="jpg"
         )
         examples.append(image)
    run.log({"examples": examples})
```


---

### <kbd>property</kbd> Image.image







---