
# 이미지

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/image.py#L64-L687' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

W&B에 로그를 기록하기 위한 이미지 형식.

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

| 인수 |  |
| :--- | :--- |
|  `data_or_path` |  (numpy 배열, 문자열, io) 이미지 데이터의 numpy 배열이나 PIL 이미지를 받습니다. 클래스는 데이터 포맷을 추론하여 변환합니다. |
|  `mode` |  (문자열) 이미지의 PIL 모드입니다. 가장 일반적인 모드는 "L", "RGB", "RGBA"입니다. 전체 설명은 https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes 에서 확인하세요. |
|  `caption` |  (문자열) 이미지의 표시 라벨입니다. |

참고: `torch.Tensor`를 `wandb.Image`로 로깅할 때, 이미지는 정규화됩니다. 이미지를 정규화하고 싶지 않다면, 텐서를 PIL 이미지로 변환해주세요.

#### 예시:

### numpy 배열에서 wandb.Image 생성하기




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

### PILImage에서 wandb.Image 생성하기




```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        pil_image = PILImage.fromarray(pixels, mode="RGB")
        image = wandb.Image(pil_image, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

### .png 대신 .jpg 로그하기 (기본값)




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

## 메서드

### `all_boxes`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/image.py#L608-L629)

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

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/image.py#L631-L635)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

### `all_masks`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/image.py#L585-L606)

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

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/image.py#L472-L484)

```python
guess_mode(
    data: "np.ndarray"
) -> str
```

np.array가 나타내는 이미지 유형을 추측합니다.

### `to_uint8`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/image.py#L486-L509)

```python
@classmethod
to_uint8(
    data: "np.ndarray"
) -> "np.ndarray"
```

이미지 데이터를 uint8로 변환합니다.

[0,1] 범위의 부동 소수점 이미지와 [0,255] 범위의 정수 이미지를 uint8로 변환하며, 필요한 경우 클리핑합니다.

| 클래스 변수 |  |
| :--- | :--- |
|  `MAX_DIMENSION`<a id="MAX_DIMENSION"></a> |  `65500` |
|  `MAX_ITEMS`<a id="MAX_ITEMS"></a> |  `108` |