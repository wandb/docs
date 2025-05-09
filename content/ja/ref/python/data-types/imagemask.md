---
title: ImageMask
menu:
  reference:
    identifier: ja-ref-python-data-types-imagemask
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/image_mask.py#L18-L247 >}}

画像マスクまたはオーバーレイを W&B にログとして記録するためにフォーマットします。

```python
ImageMask(
    val: dict,
    key: str
) -> None
```

| Args |  |
| :--- | :--- |
| `val` | (辞書) 画像を表すための以下の2つのキーのうちの1つを指定: mask_data : (2D numpy 配列) 画像内の各ピクセルに対する整数クラスラベルが含まれるマスク path : (文字列) マスクの保存された画像ファイルのパス class_labels : (整数から文字列への辞書, オプション) マスク内の整数クラスラベルを人間が読めるクラス名にマッピングします。デフォルトでは class_0, class_1, class_2 などになります。 |
| `key` | (文字列) このマスクの種類に対する読みやすい名前または ID (例: predictions, ground_truth) |

#### 例:

### 単一のマスクされた画像をログに記録

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

### テーブル内にマスクされた画像をログに記録する

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

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/image_mask.py#L219-L221)

```python
@classmethod
type_name() -> str
```

### `validate`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/image_mask.py#L223-L247)

```python
validate(
    val: dict
) -> bool
```