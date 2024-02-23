
# BoundingBoxes2D

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L16-L291' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

W&B에 로깅하기 위해 2D 바운딩 박스 오버레이로 이미지를 형식화합니다.

```python
BoundingBoxes2D(
    val: dict,
    key: str
) -> None
```

| 인수 |  |
| :--- | :--- |
|  `val` |  (사전) 다음 형식의 사전: box_data: (사전 리스트) 각 바운딩 박스마다 하나의 사전, 포함: position: (사전) 바운딩 박스의 위치와 크기, 두 가지 형식 중 하나에 유의하세요. 박스는 모두 같은 형식을 사용할 필요가 없습니다. {"minX", "minY", "maxX", "maxY"}: (사전) 박스의 상단 및 하단 경계를 정의하는 좌표 세트 (왼쪽 아래 및 오른쪽 위 모서리) {"middle", "width", "height"}: (사전) 박스의 중심과 치수를 정의하는 좌표 세트로, "middle"은 중심점을 위한 [x, y] 리스트이며 "width"와 "height"는 숫자입니다. domain: (문자열) 바운딩 박스 좌표 도메인에 대한 두 가지 옵션 중 하나 null: 기본적으로, 또는 인수가 전달되지 않은 경우, 좌표 도메인은 원본 이미지에 상대적으로 가정되며, 이 박스를 원본 이미지의 분수 또는 백분율로 표현합니다. 이는 "position" 인수로 전달된 모든 좌표 및 치수가 0과 1 사이의 부동 소수점 숫자임을 의미합니다. "pixel": (문자열 리터럴) 좌표 도메인이 픽셀 공간으로 설정됩니다. 이는 "position"으로 전달된 모든 좌표 및 치수가 이미지 치수의 한계 내에서 정수임을 의미합니다. class_id: (정수) 이 박스에 대한 클래스 레이블 id scores: (문자열에서 숫자로의 사전, 선택사항) 수치 값 (float 또는 int)에 대한 명명된 필드의 매핑, 해당 필드의 값 범위를 기반으로 UI에서 박스를 필터링하는 데 사용할 수 있습니다. box_caption: (문자열, 선택사항) UI에서 이 박스 위에 레이블 텍스트로 표시되는 문자열, 종종 클래스 레이블, 클래스 이름 및/또는 점수로 구성됩니다. class_labels: (사전, 선택사항) 가독성 있는 클래스 이름에 대한 정수 클래스 레이블의 맵 |
|  `key` |  (문자열) 이 바운딩 박스 세트의 읽을 수 있는 이름 또는 id (예: 예측값, 실측값) |

#### 예시:

### 단일 이미지에 대한 바운딩 박스 로깅

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
                    # 기본 상대적/분수 도메인에서 표현된 하나의 박스
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 1,
                    "box_caption": class_labels[1],
                    "scores": {"acc": 0.2, "loss": 1.2},
                },
                {
                    # 픽셀 도메인에서 표현된 다른 박스
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                },
                # 필요한 만큼 많은 박스를 로그로 기록하세요.
            ],
            "class_labels": class_labels,
        }
    },
)

wandb.log({"driving_scene": img})
```

### 테이블에 바운딩 박스 오버레이 로깅

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
                    # 기본 상대적/분수 도메인에서 표현된 하나의 박스
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 1,
                    "box_caption": class_labels[1],
                    "scores": {"acc": 0.2, "loss": 1.2},
                },
                {
                    # 픽셀 도메인에서 표현된 다른 박스
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                },
                # 필요한 만큼 많은 박스를 로그로 기록하세요.
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

## 메서드

### `type_name`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L215-L217)

```python
@classmethod
type_name() -> str
```

### `validate`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L219-L274)

```python
validate(
    val: dict
) -> bool
```