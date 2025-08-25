---
title: イメージ
menu:
  reference:
    identifier: ja-ref-python-data-types-image
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L65-L689 >}}

W&B に画像をログするためのフォーマット。

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

| Args |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) 画像データの numpy 配列または PIL 画像を受け付けます。クラスはデータフォーマットを推測して変換します。|
|  `mode` |  (string) 画像のための PIL モード。最も一般的なのは "L", "RGB", "RGBA" です。詳しくは https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes を参照してください。|
|  `caption` |  (string) 画像表示用のラベル。|

注意 : `wandb.Image` として `torch.Tensor` をログする際、画像は正規化されます。画像を正規化したくない場合は、テンソルを PIL Image に変換してください。

画像の正規化

PyTorch テンソルや NumPy 配列を `wandb.Image` に渡すと、`normalize=False` を設定しない限り、ピクセル値は自動的に [0, 255] の範囲に正規化されます。この正規化は、一般的な画像データ形式を処理し、適切な表示を確保するために設計されています。

正規化が適用される場合

正規化は以下に適用されます：
- **PyTorch テンソル** (形式: `(channel, height, width)`)
- **NumPy 配列** (形式: `(height, width, channel)`)

正規化は以下には適用されません：
- **PIL 画像** (そのまま渡される)
- **ファイルパス** (そのまま読み込まれる)

正規化アルゴリズム

正規化アルゴリズムは入力範囲を自動的に検出し、適切な変換を適用します：

1. **データが [0, 1] の範囲にある場合**: 値に255を掛けてuint8に変換
   ```python
   normalized_data = (data * 255).astype(np.uint8)
   ```

2. **データが [-1, 1] の範囲にある場合**: 以下の式で [0, 255] にリスケール
   ```python
   normalized_data = (255 * 0.5 * (data + 1)).astype(np.uint8)
   ```

3. **その他の範囲の場合**: 値を [0, 255] にクリップしてuint8に変換
   ```python
   normalized_data = data.clip(0, 255).astype(np.uint8)
   ```

正規化効果の例

**例1: [0, 1] 範囲のデータ**
```python
import torch
import wandb

# [0, 1] 範囲の値を持つテンソルを作成
tensor_0_1 = torch.rand(3, 64, 64)  # 0から1の間のランダム値

# すべての値に255を掛ける
image = wandb.Image(tensor_0_1, caption="[0,1]範囲から正規化")
```

**例2: [-1, 1] 範囲のデータ**
```python
import torch
import wandb

# [-1, 1] 範囲の値を持つテンソルを作成
tensor_neg1_1 = torch.rand(3, 64, 64) * 2 - 1  # -1から1の間のランダム値

# リスケール: -1 → 0, 0 → 127.5, 1 → 255
image = wandb.Image(tensor_neg1_1, caption="[-1,1]範囲から正規化")
```

**例3: PIL画像で正規化を回避**
```python
import torch
from PIL import Image as PILImage
import wandb

# [0, 1] 範囲の値を持つテンソルを作成
tensor_0_1 = torch.rand(3, 64, 64)

# PIL画像に変換して正規化を回避
pil_image = PILImage.fromarray((tensor_0_1.permute(1, 2, 0).numpy() * 255).astype('uint8'))
image = wandb.Image(pil_image, caption="正規化は適用されません")
```

**例4: normalize=False を使用**
```python
import torch
import wandb

# [0, 1] 範囲の値を持つテンソルを作成
tensor_0_1 = torch.rand(3, 64, 64)

# 正規化を無効化 - 値は [0, 255] にクリップされる
image = wandb.Image(tensor_0_1, normalize=False, caption="正規化無効")
```

ベストプラクティス

1. **一貫した結果のため**: ログする前にデータを期待される [0, 255] 範囲に前処理する
2. **正規化を回避するため**: `PILImage.fromarray()` を使用してテンソルをPIL画像に変換する
3. **デバッグのため**: `normalize=False` を使用して生の値を確認する（[0, 255] にクリップされる）
4. **精密な制御のため**: 正確なピクセル値が必要な場合はPIL画像を使用する

よくある落とし穴

- **予期しない明度**: テンソル値が [0, 1] 範囲にある場合、255倍されるため画像が非常に明るくなる
- **データ損失**: [0, 255] 範囲外の値はクリップされ、情報が失われる可能性がある
- **一貫性のない動作**: 異なる入力タイプ（テンソル vs PIL vs ファイルパス）で異なる結果が得られる可能性がある

例:

numpy 配列から wandb.Image を作成

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

.png (デフォルト) ではなく .jpg をログ

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

メソッド

`all_boxes`

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

`all_captions`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L633-L637)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

`all_masks`

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

`guess_mode`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L474-L486)

```python
guess_mode(
    data: "np.ndarray"
) -> str
```

np.array が表している画像の種類を推測します。

`to_uint8`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L488-L511)

```python
@classmethod
to_uint8(
    data: "np.ndarray"
) -> "np.ndarray"
```

画像データを uint8 に変換します。

範囲 [0,1] の浮動小数点画像と範囲 [0,255] の整数画像を必要に応じてクリッピングして uint8 に変換します。

| クラス変数 |  |
| :--- | :--- |
|  `MAX_DIMENSION`<a id="MAX_DIMENSION"></a> |  `65500` |
|  `MAX_ITEMS`<a id="MAX_ITEMS"></a> |  `108` |