---
title: 画像
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Image
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/image.py >}}




## <kbd>クラス</kbd> `Image`
W&B に画像をログするためのクラス。 

### <kbd>メソッド</kbd> `Image.__init__`

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



**引数:**
 
 - `data_or_path`:  画像データの NumPy 配列 / pytorch テンソル、PIL 画像オブジェクト、または画像ファイルへのパスを受け取ります。NumPy 配列または pytorch テンソルが与えられた場合、画像データは指定されたファイル形式で保存されます。値が [0, 255] の範囲にない、またはすべての値が [0, 1] の範囲にある場合、`normalize` が False に設定されていない限り、画素値は [0, 255] の範囲に正規化されます。 
    - pytorch テンソルは (channel, height, width) の形式である必要があります 
    - NumPy 配列は (height, width, channel) の形式である必要があります 
 - `mode`:  画像の PIL モード。よく使われるのは "L", "RGB", 
 - `"RGBA". Full explanation at https`: //pillow.readthedocs.io/en/stable/handbook/concepts.html#modes 
 - `caption`:  画像の表示用ラベル。 
 - `grouping`:  この画像のグルーピング番号。 
 - `classes`:  画像のクラス情報のリストで、バウンディングボックスや画像マスクのラベル付けに使用します。 
 - `boxes`:  画像のバウンディングボックス情報を含む 辞書。 
 - `see`:  https://docs.wandb.ai/ref/python/data-types/boundingboxes2d/ 
 - `masks`:  画像のマスク情報を含む 辞書。 
 - `see`:  https://docs.wandb.ai/ref/python/data-types/imagemask/ 
 - `file_type`:  画像を保存するファイル形式。`data_or_path` が画像ファイルへのパスである場合、このパラメータは効果がありません。 
 - `normalize`:  True の場合、画像の画素値を [0, 255] の範囲に正規化します。正規化は、`data_or_path` が numpy 配列または pytorch テンソルの場合にのみ適用されます。 



**例:**
 NumPy 配列から wandb.Image を作成 

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

PILImage から wandb.Image を作成 

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

デフォルトの .png ではなく .jpg でログ 

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(
             pixels, caption=f"random field {i}", file_type="jpg"
         )
         examples.append(image)
    run.log({"examples": examples})
``` 


---

### <kbd>プロパティ</kbd> Image.image







---