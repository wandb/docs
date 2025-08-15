---
title: 이미지
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-Image
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/image.py >}}




## <kbd>class</kbd> `Image`
W&B에 이미지를 로그하기 위한 클래스입니다.

### <kbd>method</kbd> `Image.__init__`

```python
__init__(
    data_or_path: 'ImageDataOrPathType',
    mode: Optional[str] = None,
    caption: Optional[str] = None,
    grouping: Optional[int] = None,
    classes: Optional[ForwardRef('Classes'), Sequence[dict]] = None,
    boxes: Optional[Dict[str, ForwardRef('BoundingBoxes2D')], Dict[str, dict]] = None,
    masks: Optional[Dict[str, ForwardRef('ImageMask')], Dict[str, dict]] = None,
    file_type: Optional[str] = None,
    normalize: bool = True
) → None
```

`wandb.Image` 오브젝트를 초기화합니다.



**파라미터:**

 - `data_or_path`:  NumPy array/pytorch tensor 형식의 이미지 데이터, PIL 이미지 오브젝트, 또는 이미지 파일의 경로를 입력받습니다. NumPy array나 pytorch tensor가 주어지면, 해당 이미지 데이터가 지정한 파일 형식으로 저장됩니다. 값이 [0, 255] 범위에 없거나, 모든 값이 [0, 1] 범위에 있을 경우, `normalize`가 False가 아니라면 이미지 픽셀 값이 [0, 255]로 정규화됩니다.
    - pytorch tensor는 (channel, height, width) 형식이어야 합니다.
    - NumPy array는 (height, width, channel) 형식이어야 합니다.
 - `mode`:  PIL 이미지 모드입니다. 가장 일반적으로는 "L", "RGB", "RGBA" 등이 있습니다. 자세한 설명은 https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes 를 참고하세요.
 - `caption`:  이미지 출력 시 표시할 라벨입니다.
 - `grouping`:  이미지의 그룹 번호입니다.
 - `classes`:  이미지를 위한 클래스 정보 리스트로, 바운딩 박스 및 이미지 마스크의 라벨링에 사용됩니다.
 - `boxes`:  이미지의 바운딩 박스 정보를 담고 있는 사전입니다.  
   - 자세한 정보: https://docs.wandb.ai/ref/python/data-types/boundingboxes2d/
 - `masks`:  이미지의 마스크 정보를 담고 있는 사전입니다.
   - 자세한 정보: https://docs.wandb.ai/ref/python/data-types/imagemask/
 - `file_type`:  이미지를 저장할 파일 확장자입니다. `data_or_path`가 파일 경로인 경우에는 이 파라미터가 무시됩니다.
 - `normalize`:  True로 설정하면 이미지 픽셀 값을 [0, 255] 범위로 정규화합니다. 정규화는 `data_or_path`가 numpy array 또는 pytorch tensor일 때만 적용됩니다.



**예시:**
 numpy array에서 wandb.Image 생성하기

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         # 임의의 이미지를 생성합니다.
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(pixels, caption=f"random field {i}")
         examples.append(image)
    run.log({"examples": examples})
```

PILImage로 wandb.Image 생성하기

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         # 임의의 이미지를 생성합니다.
         pixels = np.random.randint(
             low=0, high=256, size=(100, 100, 3), dtype=np.uint8
         )
         pil_image = PILImage.fromarray(pixels, mode="RGB")
         image = wandb.Image(pil_image, caption=f"random field {i}")
         examples.append(image)
    run.log({"examples": examples})
```

기본값(.png) 대신 .jpg로 저장하기

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         # 임의의 이미지를 생성합니다.
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(
             pixels, caption=f"random field {i}", file_type="jpg"
         )
         examples.append(image)
    run.log({"examples": examples})
```


---

### <kbd>property</kbd> Image.image







---