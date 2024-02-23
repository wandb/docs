
# 이미지마스크

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/helper_types/image_mask.py#L18-L233' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

W&B에 로깅하기 위한 이미지 마스크 또는 오버레이 포맷.

```python
ImageMask(
    val: dict,
    key: str
) -> None
```

| 인수 |  |
| :--- | :--- |
|  `val` |  (사전) 이미지를 나타내는 두 키 중 하나: mask_data : (2D numpy 배열) 이미지의 각 픽셀에 대한 정수 클래스 라벨이 포함된 마스크 path : (문자열) 마스크 이미지 파일이 저장된 경로 class_labels : (정수에서 문자열로의 사전, 선택 사항) 마스크 내의 정수 클래스 라벨을 읽을 수 있는 클래스 이름으로 매핑합니다. 이 값은 기본적으로 class_0, class_1, class_2 등으로 설정됩니다. |
|  `key` |  (문자열) 이 마스크 타입에 대한 읽을 수 있는 이름 또는 id (예: 예측값, ground_truth) |

#### 예시:

### 단일 마스크 이미지 로깅

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

class_labels = {0: "사람", 1: "나무", 2: "자동차", 3: "도로"}

masked_image = wandb.Image(
    image,
    masks={
        "예측값": {"mask_data": predicted_mask, "class_labels": class_labels},
        "ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels},
    },
)
wandb.log({"img_with_masks": masked_image})
```

### 테이블 내에서 마스크 이미지 로깅

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

class_labels = {0: "사람", 1: "나무", 2: "자동차", 3: "도로"}

class_set = wandb.Classes(
    [
        {"name": "사람", "id": 0},
        {"name": "나무", "id": 1},
        {"name": "자동차", "id": 2},
        {"name": "도로", "id": 3},
    ]
)

masked_image = wandb.Image(
    image,
    masks={
        "예측값": {"mask_data": predicted_mask, "class_labels": class_labels},
        "ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels},
    },
    classes=class_set,
)

table = wandb.Table(columns=["image"])
table.add_data(masked_image)
wandb.log({"random_field": table})
```

## 메서드

### `type_name`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/helper_types/image_mask.py#L205-L207)

```python
@classmethod
type_name() -> str
```

### `validate`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/helper_types/image_mask.py#L209-L233)

```python
validate(
    val: dict
) -> bool
```