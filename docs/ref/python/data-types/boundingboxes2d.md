
# BoundingBoxes2D

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L16-L293' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

画像に2Dバウンディングボックスのオーバーレイを追加して、W&Bにログを記録します。

```python
BoundingBoxes2D(
    val: dict,
    key: str
) -> None
```

| 引数 |  |
| :--- | :--- |
|  `val` |  (辞書) 以下の形式の辞書: box_data: (辞書のリスト) 各バウンディングボックスごとに1つの辞書を持つ: position: (辞書) バウンディングボックスの位置とサイズを、次の2つの形式のいずれかで Note that boxes need not all use the same format. {"minX", "minY", "maxX", "maxY"}: (辞書) ボックスの上限と下限を定義する座標のセット（左下と右上の角） {"middle", "width", "height"}: (辞書) 中心と寸法を定義する座標のセット、「middle」は中心点のリスト[x, y]、「width」と「height」は数字です domain: (string) バウンディングボックスの座標ドメインの2つのオプションのいずれか null: デフォルトで、または引数が渡されない場合、座標ドメインは元の画像に相対的であると見なされ、このボックスを元の画像の割合またはパーセンテージとして表現します。これにより、"position" 引数に渡されるすべての座標と寸法は、0から1の浮動小数点数になります。 "pixel": (文字列リテラル) 座標ドメインはピクセル空間に設定されます。これにより、"position" に渡されるすべての座標と寸法は、画像寸法の範囲内の整数になります。 class_id: (整数) このボックスのクラスラベルID scores: (文字列から数字への辞書, 任意) 名前付きフィールドを数値 (float または int) にマッピングします。対応するフィールドの値の範囲に基づいてUIでボックスをフィルタリングするために使用できます box_caption: (文字列, 任意) UIでこのボックスの上にラベルテキストとして表示する文字列。通常、クラスラベル、クラス名、および/またはスコアで構成されることが多い class_labels: (辞書, 任意) 読みやすいクラス名への整数クラスラベルのマップ |
|  `key` |  (string) バウンディングボックスのこのセットの読みやすい名前またはID（例: 予測, ground_truth） |

#### 例:

### 単一の画像にバウンディングボックスをログに記録する

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
                    # デフォルトの相対/分数ドメインで表現された1つのボックス
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
                # 必要なだけボックスをログに記録
            ],
            "class_labels": class_labels,
        }
    },
)

wandb.log({"driving_scene": img})
```

### Tableにバウンディングボックスのオーバーレイをログに記録する

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
                    # デフォルトの相対/分数ドメインで表現された1つのボックス
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
                # 必要なだけボックスをログに記録
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

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L215-L217)

```python
@classmethod
type_name() -> str
```

### `validate`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L219-L276)

```python
validate(
    val: dict
) -> bool
```