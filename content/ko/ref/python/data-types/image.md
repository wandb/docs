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

이미지 정규화

PyTorch 텐서나 NumPy 배열을 `wandb.Image`에 전달하면 `normalize=False`를 설정하지 않는 한 픽셀 값이 자동으로 [0, 255] 범위로 정규화됩니다. 이 정규화는 일반적인 이미지 데이터 형식을 처리하고 적절한 표시를 보장하기 위해 설계되었습니다.

정규화가 적용되는 경우

정규화는 다음에 적용됩니다:
- **PyTorch 텐서** (형식: `(channel, height, width)`)
- **NumPy 배열** (형식: `(height, width, channel)`)

정규화는 다음에 적용되지 않습니다:
- **PIL 이미지** (그대로 전달됨)
- **파일 경로** (그대로 로드됨)

정규화 알고리즘

정규화 알고리즘은 입력 범위를 자동으로 감지하고 적절한 변환을 적용합니다:

1. **데이터가 [0, 1] 범위에 있는 경우**: 값에 255를 곱하고 uint8로 변환
   ```python
   normalized_data = (data * 255).astype(np.uint8)
   ```

2. **데이터가 [-1, 1] 범위에 있는 경우**: 다음 공식으로 [0, 255]에 리스케일
   ```python
   normalized_data = (255 * 0.5 * (data + 1)).astype(np.uint8)
   ```

3. **기타 범위의 경우**: 값을 [0, 255]로 클립하고 uint8로 변환
   ```python
   normalized_data = data.clip(0, 255).astype(np.uint8)
   ```

정규화 효과 예시

**예시 1: [0, 1] 범위 데이터**
```python
import torch
import wandb

# [0, 1] 범위의 값을 가진 텐서 생성
tensor_0_1 = torch.rand(3, 64, 64)  # 0에서 1 사이의 랜덤 값

# 모든 값에 255를 곱함
image = wandb.Image(tensor_0_1, caption="[0,1] 범위에서 정규화")
```

**예시 2: [-1, 1] 범위 데이터**
```python
import torch
import wandb

# [-1, 1] 범위의 값을 가진 텐서 생성
tensor_neg1_1 = torch.rand(3, 64, 64) * 2 - 1  # -1에서 1 사이의 랜덤 값

# 리스케일: -1 → 0, 0 → 127.5, 1 → 255
image = wandb.Image(tensor_neg1_1, caption="[-1,1] 범위에서 정규화")
```

**예시 3: PIL 이미지로 정규화 회피**
```python
import torch
from PIL import Image as PILImage
import wandb

# [0, 1] 범위의 값을 가진 텐서 생성
tensor_0_1 = torch.rand(3, 64, 64)

# PIL 이미지로 변환하여 정규화 회피
pil_image = PILImage.fromarray((tensor_0_1.permute(1, 2, 0).numpy() * 255).astype('uint8'))
image = wandb.Image(pil_image, caption="정규화가 적용되지 않음")
```

**예시 4: normalize=False 사용**
```python
import torch
import wandb

# [0, 1] 범위의 값을 가진 텐서 생성
tensor_0_1 = torch.rand(3, 64, 64)

# 정규화 비활성화 - 값은 [0, 255]로 클립됨
image = wandb.Image(tensor_0_1, normalize=False, caption="정규화 비활성화")
```

모범 사례

1. **일관된 결과를 위해**: 로깅하기 전에 데이터를 예상되는 [0, 255] 범위로 전처리
2. **정규화를 회피하기 위해**: `PILImage.fromarray()`를 사용하여 텐서를 PIL 이미지로 변환
3. **디버깅을 위해**: `normalize=False`를 사용하여 원시 값 확인 ([0, 255]로 클립됨)
4. **정밀한 제어를 위해**: 정확한 픽셀 값이 필요한 경우 PIL 이미지 사용

일반적인 함정

- **예상치 못한 밝기**: 텐서 값이 [0, 1] 범위에 있으면 255배되어 이미지가 매우 밝아짐
- **데이터 손실**: [0, 255] 범위를 벗어나는 값은 클립되어 정보가 손실될 수 있음
- **일관성 없는 동작**: 다른 입력 유형(텐서 vs PIL vs 파일 경로)에서 다른 결과가 나올 수 있음

예시:

numpy array에서 wandb.Image 생성

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

PILImage에서 wandb.Image 생성

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

.png (기본값) 대신 .jpg 로깅

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

`all_boxes`

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

`all_captions`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L633-L637)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

`all_masks`

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

`guess_mode`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/image.py#L474-L486)

```python
guess_mode(
    data: "np.ndarray"
) -> str
```

np.array가 나타내는 이미지 유형을 추측합니다.

`to_uint8`

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
