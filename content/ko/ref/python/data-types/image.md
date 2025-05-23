---
title: Image
menu:
  reference:
    identifier: ko-ref-python-data-types-image
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L65-L689 >}}

W\&B에 로깅하기 위한 이미지 형식입니다.

```python
Image(
    data_or_path: "ImageDataOrPathType",
    mode: Optional[str] = None,
    caption: Optional[str] = None,
    grouping: Optional[int] = None,
    classes: Optional[Union['Classes', Sequence[dict]]] = None,
    boxes: Optional[Union[Dict[str, 'BoundingBoxes2D'], Dict[str, dict]]] = None,
    masks: Optional[Union[Dict[str, 'ImageMask'], Dict[str, dict]]] = None,
    file_type: Optional[str] = None
) -> None
```

| ARG |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) 이미지 데이터의 numpy array 또는 PIL 이미지를 허용합니다. 클래스는 데이터 형식을 추론하고 변환을 시도합니다. |
|  `mode` |  (string) 이미지의 PIL 모드입니다. 가장 일반적인 것은 "L", "RGB", "RGBA"입니다. 자세한 설명은 https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes 를 참조하세요. |
|  `caption` |  (string) 이미지 표시를 위한 레이블입니다. |

참고 : `torch.Tensor`를 `wandb.Image`로 로깅할 때 이미지는 정규화됩니다. 이미지를 정규화하지 않으려면 텐서를 PIL Image로 변환하십시오.

#### 예시:

### numpy array에서 wandb.Image 생성

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
        image = wandb.Image(pixels, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

### PILImage에서 wandb.Image 생성

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(
            low=0, high=256, size=(100, 100, 3), dtype=np.uint8
        )
        pil_image = PILImage.fromarray(pixels, mode="RGB")
        image = wandb.Image(pil_image, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

### .png (기본값) 대신 .jpg 로깅

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
        image = wandb.Image(pixels, caption=f"random field {i}", file_type="jpg")
        examples.append(image)
    run.log({"examples": examples})
```

| 속성 |  |
| :--- | :--- |

## Methods

### `all_boxes`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L610-L631)

```python
@classmethod
all_boxes(
    images: Sequence['Image'],
    run: "LocalRun",
    run_key: str,
    step: Union[int, str]
) -> Union[List[Optional[dict]], bool]
```

### `all_captions`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L633-L637)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

### `all_masks`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L587-L608)

```python
@classmethod
all_masks(
    images: Sequence['Image'],
    run: "LocalRun",
    run_key: str,
    step: Union[int, str]
) -> Union[List[Optional[dict]], bool]
```

### `guess_mode`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L474-L486)

```python
guess_mode(
    data: "np.ndarray"
) -> str
```

np.array가 나타내는 이미지 유형을 추측합니다.

### `to_uint8`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L488-L511)

```python
@classmethod
to_uint8(
    data: "np.ndarray"
) -> "np.ndarray"
```

이미지 데이터를 uint8로 변환합니다.

[0,1] 범위의 부동 소수점 이미지와 [0,255] 범위의 정수 이미지를 uint8로 변환하고 필요한 경우 클리핑합니다.

| 클래스 변수 |  |
| :--- | :--- |
|  `MAX_DIMENSION`<a id="MAX_DIMENSION"></a> |  `65500` |
|  `MAX_ITEMS`<a id="MAX_ITEMS"></a> |  `108` |
