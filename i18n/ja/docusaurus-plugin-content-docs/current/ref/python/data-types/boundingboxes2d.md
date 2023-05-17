# BoundingBoxes2D

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L17-L292)

W&Bにログを送信するための2Dバウンディングボックスのオーバーレイを含む画像をフォーマットします。

```python
BoundingBoxes2D(
 val: dict,
 key: str
) -> None
```

| 引数 | |
| :--- | :--- |
| `val` | (辞書型) 以下の形式の辞書型: box_data: (辞書型のリスト) それぞれのバウンディングボックスの情報を含む辞書型 position: (辞書型) バウンディングボックスの位置とサイズを2つの形式のうちの1つで表現する（すべてのボックスが同じ形式を使う必要はありません）. {"minX", "minY", "maxX", "maxY"}: (辞書型) ボックスの上限と下限（左下の角と右上の角）を示す座標のセット {"middle", "width", "height"}: (辞書型) ボックスの中心と寸法を示す座標のセット。"middle"は中心点の[x, y]リストであり、"width"と"height"は数値です domain: (文字列) バウンディングボックスの座標ドメインを2つのオプションのうちの1つにする null: デフォルトであるか、引数が渡されない場合、座標ドメインは元の画像に相対的であり、このボックスを元の画像の割合またはパーセンテージとして表現します。これは、「position」引数に渡されるすべての座標と寸法が0から1の範囲の浮動小数点数であることを意味します。 "pixel": (文字列リテラル) 座標ドメインがピクセル空間に設定されます。これは、「position」に渡されるすべての座標と寸法が画像寸法の範囲内の整数であることを意味します。 class_id: (整数) このボックスのクラスラベルID scores: (文字列と数値の辞書型, 任意) 名前が付けられたフィールドと数値（浮動小数点または整数）のマッピングで、UIでの該当フィールドの値範囲に基づくボックスのフィルタリングに使用できます box_caption: (文字列, 任意) UI上でこのボックスの上に表示されるラベルテキストとして表示される文字列で、クラスラベル、クラス名、および/またはスコアで構成されることがよくあります class_labels: (辞書型, 任意) 整数型のクラスラベルと読みやすいクラス名のマップ |
| `key` | (文字列) このバウンディングボックスのセットの読みやすい名前またはID（例: predictions, ground_truth） |

#### 例:
### 一つの画像のバウンディングボックスをログに記録

```python
import numpy as np
import wandb

wandb.init()
image = np.random.randint(low=0, high=256, size=(200, 300, 3))

class_labels = {0: "人", 1: "車", 2: "道路", 3: "建物"}

img = wandb.Image(
 image,
 boxes={
 "predictions": {
 "box_data": [
 {
 # デフォルトの相対／分数表現で一つのボックスを表す
 "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
 "class_id": 1,
 "box_caption": class_labels[1],
 "scores": {"acc": 0.2, "loss": 1.2},
 },
 {
 # ピクセル領域で表現されたもう一つのボックス
 "position": {"middle": [150, 20], "width": 68, "height": 112},
 "domain": "pixel",
 "class_id": 3,
 "box_caption": "建物",
 "scores": {"acc": 0.5, "loss": 0.7},
 },
 # 必要なだけボックスを記録
 ],
 "class_labels": class_labels,
 }
 },
)

```
wandb.log({"driving_scene": img})
```

### テーブルにバウンディングボックスのオーバーレイをログする

```python
import numpy as np
import wandb

wandb.init()
image = np.random.randint(low=0, high=256, size=(200, 300, 3))

class_labels = {0: "人", 1: "車", 2: "道路", 3: "建物"}

class_set = wandb.Classes(
 [
 {"name": "人", "id": 0},
 {"name": "車", "id": 1},
 {"name": "道路", "id": 2},
 {"name": "建物", "id": 3},
 ]
)

img = wandb.Image(
 image,
 boxes={
 "predictions": {
 "box_data": [
 {
 # デフォルトの相対/分数ドメインで表される1つのボックス
 "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
 "class_id": 1,
 "box_caption": class_labels[1],
 "scores": {"acc": 0.2, "loss": 1.2},
 },
 {
 # ピクセルドメインで表されるもう1つのボックス
 "position": {"middle": [150, 20], "width": 68, "height": 112},
 "domain": "pixel",
 "class_id": 3,
 "box_caption": "建物",
 "scores": {"acc": 0.5, "loss": 0.7},
 },
 # 必要なだけのボックスをログする
 ],
 "class_labels": class_labels,
 }
 },
 classes=class_set,
)
```

table = wandb.Table(columns=["image"])
table.add_data(img)
wandb.log({"driving_scene": table})
```


## メソッド

### `type_name`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L216-L218)

```python
@classmethod
type_name() -> str
```




### `validate`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L220-L275)

```python
validate(
 val: dict
) -> bool
```