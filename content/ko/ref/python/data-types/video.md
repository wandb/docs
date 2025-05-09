---
title: Video
menu:
  reference:
    identifier: ko-ref-python-data-types-video
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/video.py#L49-L251 >}}

W&B에 로깅하기 위한 비디오 포맷입니다.

```python
Video(
    data_or_path: Union['np.ndarray', str, 'TextIO', 'BytesIO'],
    caption: Optional[str] = None,
    fps: Optional[int] = None,
    format: Optional[str] = None
)
```

| Args |  |
| :--- | :--- |
|  `data_or_path` | (numpy array, string, io) 비디오는 파일 경로 또는 io 오브젝트로 초기화할 수 있습니다. 포맷은 "gif", "mp4", "webm" 또는 "ogg"이어야 합니다. 포맷은 format 인수로 지정해야 합니다. 비디오는 numpy 텐서로 초기화할 수 있습니다. numpy 텐서는 4차원 또는 5차원이어야 합니다. 채널은 (시간, 채널, 높이, 너비) 또는 (배치, 시간, 채널, 높이, 너비)여야 합니다. |
|  `caption` | (string) 표시에 사용될 비디오와 관련된 캡션입니다. |
|  `fps` | (int) 원시 비디오 프레임을 인코딩할 때 사용할 프레임 속도입니다. 기본값은 4입니다. 이 파라미터는 data_or_path가 string 또는 bytes인 경우 영향을 미치지 않습니다. |
|  `format` | (string) 비디오 포맷입니다. 경로 또는 io 오브젝트로 초기화하는 경우 필요합니다. |

#### Examples:

### numpy array를 비디오로 로그합니다.

```python
import numpy as np
import wandb

run = wandb.init()
# axes are (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

## Methods

### `encode`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/video.py#L140-L179)

```python
encode(
    fps: int = 4
) -> None
```

| Class Variables |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |
