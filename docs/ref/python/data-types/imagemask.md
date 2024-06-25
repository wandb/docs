
# ImageMask

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/helper_types/image_mask.py#L18-L233' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

画像マスクやオーバーレイをW&Bにログするためにフォーマットします。

```python
ImageMask(
    val: dict,
    key: str
) -> None
```

| 引数 |  |
| :--- | :--- |
|  `val` |  (辞書) 画像を表すための次の2つのキーのうちの1つ: mask_data : (2D numpy配列) 画像内の各ピクセルに対する整数クラスラベルを含むマスク path : (文字列) マスクの保存された画像ファイルのパス class_labels : (整数から文字列への辞書, オプション) マスク内の整数クラスラベルを読み取り可能なクラス名にマッピングする。このデフォルトはclass_0, class_1, class_2などになります。 |
|  `key` |  (文字列) このマスクタイプの読み取り可能な名前またはID（例：予測値, ground_truth） |

#### 例:

### 単一のマスク画像のログ

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

class_labels = {0: "person", 1: "tree", 2: "car", 3: "road"}

masked_image = wandb.Image(
    image,
    masks={
        "predictions": {"mask_data": predicted_mask, "class_labels": class_labels},
        "ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels},
    },
)
wandb.log({"img_with_masks": masked_image})
```

### Masked imageをTable内にログする

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

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/helper_types/image_mask.py#L205-L207)

```python
@classmethod
type_name() -> str
```

### `validate`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/helper_types/image_mask.py#L209-L233)

```python
validate(
    val: dict
) -> bool
```