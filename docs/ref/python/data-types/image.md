# Image

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/image.py#L64-L689' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

이미지를 W&B에 로그하도록 포맷합니다.

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
|  `data_or_path` |  (numpy array, string, io) 이미지 데이터의 numpy array 또는 PIL 이미지를 허용합니다. 클래스는 데이터 포맷을 추론하여 변환을 시도합니다. |
|  `mode` |  (string) 이미지의 PIL 모드입니다. 가장 흔한 것은 "L", "RGB", "RGBA"입니다. 전체 설명은 https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes에서 확인할 수 있습니다. |
|  `caption` |  (string) 이미지의 표시를 위한 레이블입니다. |

Note : `torch.Tensor`를 `wandb.Image`로 로그할 때, 이미지는 정규화됩니다. 이미지를 정규화하지 않으려면, 텐서를 PIL Image로 변환하세요.

#### 예제:

### numpy array로부터 wandb.Image 생성하기

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

### PILImage로부터 wandb.Image 생성하기

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

### .png (기본값) 대신 .jpg 로그하기

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

## 메소드

### `all_boxes`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/image.py#L610-L631)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/image.py#L633-L637)

```python
@classmethod
all_captions(
    images: Sequence['Media']
) -> Union[bool, Sequence[Optional[str]]]
```

### `all_masks`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/image.py#L587-L608)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/image.py#L474-L486)

```python
guess_mode(
    data: "np.ndarray"
) -> str
```

np.array가 표현하는 이미지 유형을 추측합니다.

### `to_uint8`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/image.py#L488-L511)

```python
@classmethod
to_uint8(
    data: "np.ndarray"
) -> "np.ndarray"
```

이미지 데이터를 uint8로 변환합니다.

범위 [0,1]의 부동 소수점 이미지와 범위 [0,255]의 정수 이미지를 uint8로 변환하며, 필요한 경우 잘라냅니다.

| 클래스 변수 |  |
| :--- | :--- |
|  `MAX_DIMENSION`<a id="MAX_DIMENSION"></a> |  `65500` |
|  `MAX_ITEMS`<a id="MAX_ITEMS"></a> |  `108` |