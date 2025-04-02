---
title: ImageMask
menu:
  reference:
    identifier: ja-ref-python-data-types-imagemask
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/image_mask.py#L18-L247 >}}

W&B に ログ を記録するための画像マスクまたはオーバーレイのフォーマット。

```python
ImageMask(
    val: dict,
    key: str
) -> None
```

| arg |  |
| :--- | :--- |
|  `val` |  (dictionary) 画像を表す次の2つの キー のいずれか: mask_data : (2D numpy 配列) 画像内の各ピクセルの整数クラスラベルを含むマスク path : (string) マスクの保存された画像ファイルへのパス class_labels : (integer から string への dictionary, optional) マスク内の integer クラスラベルから読み取り可能なクラス名へのマッピング。これらはデフォルトで class_0, class_1, class_2 などになります。 |
|  `key` |  (string) このマスクタイプの読み取り可能な名前または ID (例: predictions, ground_truth) |

#### 例:

### マスクされた単一の画像を ログ に記録する

```python
import numpy as np
import wandb

run = wandb.init()
image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
predicted_mask = np.empty((100, 100), dtype=np.uint8)
ground_truth_mask = np.empty((100, 100), dtype=np.uint8)

predicted_mask[:50, :50] = 0
predicted_mask[50:, :50] = 1
predicted_mask[:50, 50:] = 2
predicted_mask[50:, 50:] = 3

ground_truth_mask[:25, :25] = 0
ground_truth_mask[25:, :25] = 1
ground_truth_mask[:25, 25:] = 2
ground_truth_mask[25:, 25:] = 3

class_labels = {0: "person", 1: "tree", 2: "car", 3: "road"}

masked_image = wandb.Image(
    image,
    masks={
        "predictions": {
            "mask_data": predicted_mask,
            "class_labels": class_labels,
        },
        "ground_truth": {
            "mask_data": ground_truth_mask,
            "class_labels": class_labels,
        },
    },
)
run.log({"img_with_masks": masked_image})
```

### Table 内でマスクされた画像を ログ に記録する

```python
import numpy as np
import wandb

run = wandb.init()
image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
predicted_mask = np.empty((100, 100), dtype=np.uint8)
ground_truth_mask = np.empty((100, 100), dtype=np.uint8)

predicted_mask[:50, :50] = 0
predicted_mask[50:, :50] = 1
predicted_mask[:50, 50:] = 2
predicted_mask[50:, 50:] = 3

ground_truth_mask[:25, :25] = 0
ground_truth_mask[25:, :25] = 1
ground_truth_mask[:25, 25:] = 2
ground_truth_mask[25:, 25:] = 3

class_labels = {0: "person", 1: "tree", 2: "car", 3: "road"}

class_set = wandb.Classes(
    [
        {"name": "person", "id": 0},
        {"name": "tree", "id": 1},
        {"name": "car", "id": 2},
        {"name": "road", "id": 3},
    ]
)

masked_image = wandb.Image(
    image,
    masks={
        "predictions": {
            "mask_data": predicted_mask,
            "class_labels": class_labels,
        },
        "ground_truth": {
            "mask_data": ground_truth_mask,
            "class_labels": class_labels,
        },
    },
    classes=class_set,
)

table = wandb.Table(columns=["image"])
table.add_data(masked_image)
run.log({"random_field": table})
```

## メソッド

### `type_name`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/image_mask.py#L219-L221)

```python
@classmethod
type_name() -> str
```

### `validate`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/image_mask.py#L223-L247)

```python
validate(
    val: dict
) -> bool
```