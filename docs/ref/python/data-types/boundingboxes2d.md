# BoundingBoxes2D

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L16-L293' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

画像に2Dバウンディングボックスをオーバーレイし、W&Bにログを取るフォーマットです。

```python
BoundingBoxes2D(
    val: dict,
    key: str
) -> None
```

| 引数 |  |
| :--- | :--- |
|  `val` |  (辞書) 以下の形式の辞書: box_data: (辞書のリスト) 各バウンディングボックスのための辞書: position: (辞書) バウンディングボックスの位置とサイズ、以下の2つの形式のどちらかを使用。ボックスは同じ形式を使用する必要はありません。 {"minX", "minY", "maxX", "maxY"}: (辞書) ボックスの上下の境界を定義する座標セット（左下と右上のコーナー） {"middle", "width", "height"}: (辞書) ボックスの中心と寸法を定義する座標セットで、中心点の「middle」は[x, y]のリスト、「width」と「height」は数値です。 domain: (文字列) バウンディングボックスの座標ドメインの2つのオプションのうちの1つ null: デフォルトでは、または引数が渡されない場合、座標ドメインは元の画像に対する相対座標として扱われます。このため、"position"引数に渡されるすべての座標と寸法は0から1の範囲の浮動小数点数になります。 "pixel": (文字列リテラル) 座標ドメインがピクセル空間として設定されます。これにより、"position"に渡されるすべての座標と寸法は画像寸法内の整数になります。 class_id: (整数) このボックスのクラスラベルID scores: (文字列から数値への辞書, オプショナル) フィールド名と数値（浮動小数点数または整数）をマッピングする辞書。UIで対応するフィールドの値の範囲に基づいてボックスをフィルタリングするために使用できます box_caption: (文字列, オプショナル) UIでこのボックスの上にラベルテキストとして表示される文字列で、クラスラベル、クラス名、および/またはスコアで構成されることがよくあります class_labels: (辞書, オプショナル) 整数のクラスラベルを読みやすいクラス名にマッピングする辞書 |
|  `key` |  (文字列) このバウンディングボックスセットの読みやすい名前またはID (例: predictions, ground_truth) |

#### 例:

### 単一の画像に対するバウンディングボックスのログ

```python
import numpy as np
import wandb

wandb.init()
image = np.random.randint(low=0, high=256, size=(200, 300, 3))

class_labels = {0: "person", 1: "car", 2: "road", 3: "building"}

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # デフォルトの相対/割合ドメインで表現された1つのボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 1,
                    "box_caption": class_labels[1],
                    "scores": {"acc": 0.2, "loss": 1.2},
                },
                {
                    # ピクセルドメインで表現された別のボックス
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                },
                # 必要なだけ多くのボックスをログする
            ],
            "class_labels": class_labels,
        }
    },
)

wandb.log({"driving_scene": img})
```

### Tableにバウンディングボックスのオーバーレイをログ

```python
import numpy as np
import wandb

wandb.init()
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
                    # デフォルトの相対/割合ドメインで表現された1つのボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 1,
                    "box_caption": class_labels[1],
                    "scores": {"acc": 0.2, "loss": 1.2},
                },
                {
                    # ピクセルドメインで表現された別のボックス
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                },
                # 必要なだけ多くのボックスをログする
            ],
            "class_labels": class_labels,
        }
    },
    classes=class_set,
)

table = wandb.Table(columns=["image"])
table.add_data(img)
wandb.log({"driving_scene": table})
```

## メソッド

### `type_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L215-L217)

```python
@classmethod
type_name() -> str
```

### `validate`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L219-L276)

```python
validate(
    val: dict
) -> bool
```