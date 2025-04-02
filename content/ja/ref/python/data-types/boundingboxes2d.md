---
title: BoundingBoxes2D
menu:
  reference:
    identifier: ja-ref-python-data-types-boundingboxes2d
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L16-L313 >}}

W&B にログを記録するために、2D 境界ボックスのオーバーレイで画像をフォーマットします。

```python
BoundingBoxes2D(
    val: dict,
    key: str
) -> None
```

| arg |  |
| :--- | :--- |
|  `val` |  (辞書) 次の形式の辞書: box_data: (辞書のリスト) 各境界ボックスに対して1つの辞書を含みます: position: (辞書) 境界ボックスの位置とサイズ。次の2つの形式のいずれかです。ボックスはすべて同じ形式を使用する必要はありません。{"minX", "minY", "maxX", "maxY"}: (辞書) ボックスの上限と下限を定義する座標のセット (左下隅と右上隅) {"middle", "width", "height"}: (辞書) ボックスの中心と寸法を定義する座標のセット。「middle」は中心点のリスト [x, y] で、「width」と「height」は数値です。domain: (文字列) 境界ボックスの座標ドメインの2つのオプションのいずれか: null: デフォルト、または引数が渡されない場合、座標ドメインは元の画像に対する相対的なものと見なされ、このボックスを元の画像の分数またはパーセンテージとして表現します。これは、「position」引数に渡されるすべての座標と寸法が0から1の間の浮動小数点数であることを意味します。「pixel」: (文字列リテラル) 座標ドメインはピクセル空間に設定されます。これは、「position」に渡されるすべての座標と寸法が画像寸法の範囲内の整数であることを意味します。class_id: (整数) このボックスのクラスラベルID scores: (文字列から数値への辞書、オプション) 名前付きフィールドから数値 (floatまたはint) へのマッピング。対応するフィールドの値の範囲に基づいてUIでボックスをフィルタリングするために使用できます。box_caption: (文字列、オプション) UIでこのボックスの上にラベルテキストとして表示される文字列。多くの場合、クラスラベル、クラス名、またはスコアで構成されます。class_labels: (辞書、オプション) 整数のクラスラベルから読み取り可能なクラス名へのマップ |
|  `key` |  (文字列) この境界ボックスセットの読みやすい名前またはID (例: predictions、ground_truth) |

#### 例:

### 単一の画像の境界ボックスをログに記録する

```python
import numpy as np
import wandb

run = wandb.init()
image = np.random.randint(low=0, high=256, size=(200, 300, 3))

class_labels = {0: "person", 1: "car", 2: "road", 3: "building"}

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # 1つのボックスは、デフォルトの相対/小数ドメインで表現されます
                    "position": {
                        "minX": 0.1,
                        "maxX": 0.2,
                        "minY": 0.3,
                        "maxY": 0.4,
                    },
                    "class_id": 1,
                    "box_caption": class_labels[1],
                    "scores": {"acc": 0.2, "loss": 1.2},
                },
                {
                    # 別のボックスは、ピクセル ドメインで表現されます
                    "position": {
                        "middle": [150, 20],
                        "width": 68,
                        "height": 112,
                    },
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                },
                # 必要に応じて、できるだけ多くのボックスをログに記録します
            ],
            "class_labels": class_labels,
        }
    },
)

run.log({"driving_scene": img})
```

### Table に境界ボックスのオーバーレイをログに記録する

```python
import numpy as np
import wandb

run = wandb.init()
image = np.random.randint(low=0, high=256, size=(200, 300, 3))

class_labels = {0: "person", 1: "car", 2: "road", 3: "building"}

class_set = wandb.Classes(
    [
        {"name": "person", "id": 0},
        {"name": "car", "id": 1},
        {"name": "road", "id": 2},
        {"name": "building", "id": 3},
    ]
)

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # 1つのボックスは、デフォルトの相対/小数ドメインで表現されます
                    "position": {
                        "minX": 0.1,
                        "maxX": 0.2,
                        "minY": 0.3,
                        "maxY": 0.4,
                    },
                    "class_id": 1,
                    "box_caption": class_labels[1],
                    "scores": {"acc": 0.2, "loss": 1.2},
                },
                {
                    # 別のボックスは、ピクセル ドメインで表現されます
                    "position": {
                        "middle": [150, 20],
                        "width": 68,
                        "height": 112,
                    },
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                },
                # 必要に応じて、できるだけ多くのボックスをログに記録します
            ],
            "class_labels": class_labels,
        }
    },
    classes=class_set,
)

table = wandb.Table(columns=["image"])
table.add_data(img)
run.log({"driving_scene": table})
```

## メソッド

### `type_name`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L233-L235)

```python
@classmethod
type_name() -> str
```

### `validate`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L237-L294)

```python
validate(
    val: dict
) -> bool
```
