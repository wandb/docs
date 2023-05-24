# ImageMask

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/helper_types/image_mask.py#L19-L234)

W&Bにログを記録するための画像マスクやオーバーレイのフォーマット。

```python
ImageMask(
 val: dict,
 key: str
) -> None
```

| 引数 |  |
| :--- | :--- |
| `val` | (辞書) 画像を表す以下の2つのキーのいずれか: mask_data : (2D numpy配列) 画像の各ピクセルに対する整数クラスラベルが含まれるマスク path : (文字列) マスクの保存された画像ファイルへのパス class_labels : (整数から文字列への辞書, オプション) マスク内の整数クラスラベルを読みやすいクラス名にマッピング。デフォルトではclass_0, class_1, class_2などになります。 |
| `key` | (文字列) このマスクタイプの読みやすい名前またはID（例：predictions、ground_truth） |

#### 例:
### マスク付きの単一画像のログ

```python
import numpy as np
import wandb

wandb.init()
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

class_labels = {0: "人", 1: "木", 2: "車", 3: "道路"}

masked_image = wandb.Image(
 image,
 masks={
 "predictions": {"mask_data": predicted_mask, "class_labels": class_labels},
 "ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels},
 },
)
wandb.log({"img_with_masks": masked_image})
```
### テーブル内のマスクされた画像をログに記録する

```python
import numpy as np
import wandb

wandb.init()
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

class_labels = {0: "人", 1: "木", 2: "車", 3: "道"}

class_set = wandb.Classes(
 [
 {"name": "人", "id": 0},
 {"name": "木", "id": 1},
 {"name": "車", "id": 2},
 {"name": "道", "id": 3},
 ]
)```

masked_image = wandb.Image(
 image,
 masks={
 "predictions": {"mask_data": predicted_mask, "class_labels": class_labels},
 "ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels},
 },
 classes=class_set,
)

table = wandb.Table(columns=["image"])
table.add_data(masked_image)
wandb.log({"random_field": table})
```

## メソッド

### `type_name`

[ソースを表示する](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/helper_types/image_mask.py#L206-L208)

```python
@classmethod
type_name() -> str
```

### `validate`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/helper_types/image_mask.py#L210-L234)

```python
validate(
 val: dict
) -> bool
```