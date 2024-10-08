# ImageMask

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/helper_types/image_mask.py#L18-L235' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

이미지 마스크 또는 오버레이를 W&B에 로그하기 위한 형식입니다.

```python
ImageMask(
    val: dict,
    key: str
) -> None
```

| 인수 |  |
| :--- | :--- |
|  `val` |  (사전) 이미지를 나타내기 위한 두 가지 키 중 하나: mask_data : (2D numpy array) 각 픽셀에 대한 정수 클래스 레이블을 포함하는 마스크입니다. image path : (문자열) 마스크의 저장된 이미지 파일 경로입니다. class_labels : (정수에서 문자열로의 사전, 선택사항) 마스크의 정수 클래스 레이블을 읽기 쉬운 클래스 이름으로 매핑합니다. 이는 기본적으로 class_0, class_1, class_2 등으로 설정됩니다. |
|  `key` |  (문자열) 이 마스크 유형에 대한 읽기 쉬운 이름 또는 ID입니다 (예: predictions, ground_truth) |

#### 예시:

### 단일 마스크 이미지를 로그하기

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

### 테이블 안에 마스크 이미지를 로그하기

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

## 메소드

### `type_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/helper_types/image_mask.py#L207-L209)

```python
@classmethod
type_name() -> str
```

### `validate`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/helper_types/image_mask.py#L211-L235)

```python
validate(
    val: dict
) -> bool
```