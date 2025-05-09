---
title: ImageMask
menu:
  reference:
    identifier: ko-ref-python-data-types-imagemask
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/image_mask.py#L18-L247 >}}

W\&B에 로깅하기 위한 이미지 마스크 또는 오버레이를 포맷합니다.

```python
ImageMask(
    val: dict,
    key: str
) -> None
```

| ARG |  |
| :--- | :--- |
|  `val` |  (dictionary) 이미지를 나타내는 다음 두 키 중 하나: mask_data : (2D numpy array) 이미지의 각 픽셀에 대한 정수 클래스 레이블을 포함하는 마스크 path : (string) 마스크의 저장된 이미지 파일 경로 class_labels : (정수-문자열 사전, 선택 사항) 마스크의 정수 클래스 레이블을 읽을 수 있는 클래스 이름에 매핑합니다. 기본적으로 class_0, class_1, class_2 등으로 설정됩니다. |
|  `key` |  (string) 이 마스크 유형의 읽을 수 있는 이름 또는 ID (예: 예측값, ground_truth) |

#### 예시:

### 단일 마스크 이미지 로깅

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

### 테이블 내부에 마스크된 이미지 로그

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

## Methods

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
