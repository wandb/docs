---
title: BoundingBoxes2D
menu:
  reference:
    identifier: ko-ref-python-data-types-boundingboxes2d
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L16-L313 >}}

W&B에 로깅하기 위해 2D 바운딩 박스 오버레이로 이미지 형식을 지정합니다.

```python
BoundingBoxes2D(
    val: dict,
    key: str
) -> None
```

| ARG |  |
| :--- | :--- |
|  `val` |  (사전) 다음과 같은 형태의 사전: box_data: (사전 목록) 각 바운딩 박스에 대한 사전 하나, 다음을 포함합니다: position: (사전) 바운딩 박스의 위치 및 크기 (두 가지 형식 중 하나) 상자들은 모두 동일한 형식을 사용할 필요는 없습니다. {"minX", "minY", "maxX", "maxY"}: (사전) 상자의 상한 및 하한을 정의하는 좌표 집합 (좌측 하단 및 우측 상단 모서리) {"middle", "width", "height"}: (사전) 상자의 중심과 크기를 정의하는 좌표 집합. 여기서 "middle"은 중심점을 나타내는 목록 [x, y]이고 "width" 및 "height"는 숫자입니다. domain: (문자열) 바운딩 박스 좌표 도메인에 대한 두 가지 옵션 중 하나 null: 기본적으로 또는 인수가 전달되지 않은 경우 좌표 도메인은 원래 이미지를 기준으로 하며, 이 상자를 원래 이미지의 분수 또는 백분율로 표현합니다. 즉, "position" 인수로 전달되는 모든 좌표 및 크기는 0과 1 사이의 부동 소수점 숫자입니다. "pixel": (문자열 리터럴) 좌표 도메인이 픽셀 공간으로 설정됩니다. 즉, "position"으로 전달되는 모든 좌표 및 크기는 이미지 크기의 범위 내에 있는 정수입니다. class_id: (정수) 이 상자에 대한 클래스 레이블 ID scores: (문자열-숫자 사전, 선택 사항) 이름이 지정된 필드를 숫자 값(float 또는 int)에 매핑합니다. 해당 필드의 값 범위에 따라 UI에서 상자를 필터링하는 데 사용할 수 있습니다. box_caption: (문자열, 선택 사항) UI에서 이 상자 위에 레이블 텍스트로 표시될 문자열입니다. 종종 클래스 레이블, 클래스 이름 및/또는 점수로 구성됩니다. class_labels: (사전, 선택 사항) 정수 클래스 레이블을 사람이 읽을 수 있는 클래스 이름에 매핑하는 맵입니다. |
|  `key` |  (문자열) 이 바운딩 박스 세트에 대한 사람이 읽을 수 있는 이름 또는 ID (예: 예측, ground_truth) |

#### 예시:

### 단일 이미지에 대한 바운딩 박스 로깅

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
                    # 기본 상대/분수 도메인으로 표현된 상자 하나
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
                    # 픽셀 도메인으로 표현된 다른 상자
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
                # 필요한 만큼 상자를 로깅합니다.
            ],
            "class_labels": class_labels,
        }
    },
)

run.log({"driving_scene": img})
```

### 테이블에 바운딩 박스 오버레이 로깅

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
                    # 기본 상대/분수 도메인으로 표현된 상자 하나
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
                    # 픽셀 도메인으로 표현된 다른 상자
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
                # 필요한 만큼 상자를 로깅합니다.
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

## 메소드

### `type_name`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L233-L235)

```python
@classmethod
type_name() -> str
```

### `validate`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L237-L294)

```python
validate(
    val: dict
) -> bool
```