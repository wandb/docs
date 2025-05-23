---
title: BoundingBoxes2D
menu:
  reference:
    identifier: ja-ref-python-data-types-boundingboxes2d
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L16-L313 >}}

W&B に ログ するために 2D バウンディング ボックス オーバーレイを使用して画像をフォーマットします。

```python
BoundingBoxes2D(
    val: dict,
    key: str
) -> None
```

| 引数 |  |
| :--- | :--- |
|  `val` |  (辞書) 以下の形式の辞書: box_data: (辞書のリスト) 各バウンディングボックスに対して 1 つの辞書を含む: position: (辞書) 2 つの形式のいずれかでバウンディングボックスの位置とサイズを指定します。ボックスが同じ形式を使用する必要はないことに注意してください。 {"minX", "minY", "maxX", "maxY"}: (辞書) ボックスの上限と下限を定義する座標のセット (左下と右上のコーナー) {"middle", "width", "height"}: (辞書) 中心点のリスト [x, y] としての "middle"、および数値としての "width" と "height" のボックスの中心と寸法を定義する座標のセット domain: (文字列) バウンディング ボックス座標ドメインの 2 つのオプション null: デフォルトでは、または引数が渡されない場合、座標ドメインは元の画像に対して相対的であると見なされ、このボックスを元の画像の分数またはパーセンテージとして表します。これは、「position」引数に渡されるすべての座標と寸法が 0 から 1 までの浮動小数点数であることを意味します。 "pixel": (文字列リテラル) 座標ドメインはピクセル空間に設定されます。これは、"position" 引数に渡されるすべての座標と寸法が画像寸法の範囲内の整数であることを意味します。 class_id: (整数) このボックスのクラスラベル id scores: (文字列から数値への辞書, オプション) 名前付きフィールドを数値 (float または int) にマッピングしたもので、対応するフィルター範囲内で UI 内のボックスをフィルタリングするために使用できます フィールド box_caption: (文字列, オプション) UI でこのボックスの上のラベルテキストとして表示される文字列、クラスラベル、クラス名、および/またはスコアで構成されることがよくあります class_labels: (辞書, オプション) 読み取り可能なクラス名への整数クラスラベルのマップ |
|  `key` |  (文字列) このバウンディングボックスのセットの読み取り可能な名前または ID (例: 予測, ground_truth) |

#### 例:

### 単一の画像に対するバウンディングボックスをログする

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
                    # デフォルトの相対/比率ドメインで表現された 1 つのボックス
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
                    # ピクセルドメインで表現された別のボックス
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
                # 必要に応じてログに記録するボックスを追加
            ],
            "class_labels": class_labels,
        }
    },
)

run.log({"driving_scene": img})
```

### テーブルにバウンディングボックスオーバーレイをログする

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
                    # デフォルトの相対/比率ドメインで表現された 1 つのボックス
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
                    # ピクセルドメインで表現された別のボックス
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
                # 必要に応じてログに記録するボックスを追加
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