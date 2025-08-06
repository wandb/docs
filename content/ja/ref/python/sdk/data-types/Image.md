---
title: 画像
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Image
object_type: python_sdk_data_type
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
 
 - `data_or_path`:  画像データの NumPy 配列/pytorch テンソル、PIL 画像オブジェクト、または画像ファイルへのパスを受け付けます。NumPy 配列または pytorch テンソルが指定された場合、画像データは指定したファイルタイプで保存されます。値が [0, 255] の範囲にない場合や全ての値が [0, 1] の範囲内である場合、`normalize` が False でない限り、画像のピクセル値は [0, 255] の範囲に正規化されます。
    - pytorch テンソルは (channel, height, width) の形式である必要があります
    - NumPy 配列は (height, width, channel) の形式である必要があります
 - `mode`:  PIL の画像モード。"L"、"RGB"、"RGBA" などが一般的です。詳細は https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes をご覧ください。
 - `caption`:  画像の表示用ラベル。
 - `grouping`:  画像のグループ番号。
 - `classes`:  画像に対するクラス情報のリスト。バウンディングボックスや画像マスクのラベル付けに使用します。
 - `boxes`:  画像のバウンディングボックス情報を含む辞書。
 - `see`:  https://docs.wandb.ai/ref/python/data-types/boundingboxes2d/
 - `masks`:  画像のマスク情報を含む辞書。
 - `see`:  https://docs.wandb.ai/ref/python/data-types/imagemask/
 - `file_type`:  画像を保存するファイルタイプ。data_or_path が画像ファイルのパスである場合はこのパラメータは無効です。
 - `normalize`:  True の場合、画像のピクセル値を [0, 255] の範囲に正規化します。正規化は data_or_path が numpy 配列または pytorch テンソルの時のみ適用されます。



**使用例:**
 NumPy 配列から wandb.Image を作成

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         # ランダムな画像データを作成
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         # wandb.Image オブジェクトを作成し、キャプションを付ける
         image = wandb.Image(pixels, caption=f"random field {i}")
         examples.append(image)
    # ログを作成
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
         # ランダムな画像データを作成（uint8 型）
         pixels = np.random.randint(
             low=0, high=256, size=(100, 100, 3), dtype=np.uint8
         )
         # PIL 画像を作成
         pil_image = PILImage.fromarray(pixels, mode="RGB")
         # wandb.Image オブジェクトを作成し、キャプションを付ける
         image = wandb.Image(pil_image, caption=f"random field {i}")
         examples.append(image)
    # ログを作成
    run.log({"examples": examples})
```

デフォルトの .png ではなく .jpg としてログする

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         # ランダムな画像データを作成
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         # .jpg 形式で wandb.Image オブジェクトを作成
         image = wandb.Image(
             pixels, caption=f"random field {i}", file_type="jpg"
         )
         examples.append(image)
    # ログを作成
    run.log({"examples": examples})
```


---

### <kbd>property</kbd> Image.image







---