---
title: 비디오
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-Video
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/video.py >}}




## <kbd>class</kbd> `Video`
W&B에 비디오를 로그하기 위한 클래스입니다.

### <kbd>method</kbd> `Video.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, ForwardRef('np.ndarray'), ForwardRef('TextIO'), ForwardRef('BytesIO')],
    caption: Optional[str] = None,
    fps: Optional[int] = None,
    format: Optional[Literal['gif', 'mp4', 'webm', 'ogg']] = None
)
```

W&B Video 오브젝트를 초기화합니다.



**인수:**
 
 - `data_or_path`:  비디오는 파일 경로나 io 오브젝트로 초기화할 수 있습니다. 또한 numpy 텐서로도 초기화할 수 있습니다. numpy 텐서는 반드시 4차원 또는 5차원이어야 합니다. 차원은 (프레임 수, 채널, 높이, 너비) 또는 (배치, 프레임 수, 채널, 높이, 너비) 형태여야 합니다. numpy array 또는 io 오브젝트로 초기화하는 경우 `format` 파라미터를 반드시 지정해야 합니다.
 - `caption`:  비디오와 함께 표시할 캡션입니다.
 - `fps`:  원시 비디오 프레임을 인코딩할 때 사용할 프레임 레이트입니다. 기본값은 4입니다. 이 파라미터는 data_or_path가 문자열 또는 바이트인 경우에는 적용되지 않습니다.
 - `format`:  비디오 포맷입니다. numpy array 또는 io 오브젝트로 초기화할 때 반드시 필요합니다. 이 파라미터는 비디오 데이터를 인코딩할 포맷을 결정하는 데 사용되며, 허용되는 값은 "gif", "mp4", "webm", "ogg"입니다. 값을 지정하지 않으면 기본값인 "gif"가 사용됩니다.



**예시:**
 numpy array를 비디오로 로그하기

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 축은 (프레임 수, 채널, 높이, 너비)입니다.
    frames = np.random.randint(
         low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8
    )
    run.log({"video": wandb.Video(frames, format="mp4", fps=4)})
``` 




---