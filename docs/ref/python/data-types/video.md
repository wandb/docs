# Video

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/video.py#L48-L239' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&B에 로그할 비디오를 포맷합니다.

```python
Video(
    data_or_path: Union['np.ndarray', str, 'TextIO', 'BytesIO'],
    caption: Optional[str] = None,
    fps: int = 4,
    format: Optional[str] = None
)
```

| 인수 |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) 비디오는 파일의 경로나 io 오브젝트로 초기화될 수 있습니다. 형식은 "gif", "mp4", "webm" 또는 "ogg"여야 합니다. 형식은 format 인수로 지정해야 합니다. 비디오는 numpy tensor로 초기화될 수 있습니다. numpy tensor는 4차원 또는 5차원이어야 합니다. 채널은 (time, channel, height, width) 또는 (batch, time, channel, height width)여야 합니다. |
|  `caption` |  (string) 비디오와 연관된 캡션으로 표시됩니다. |
|  `fps` |  (int) 비디오의 초당 프레임 수입니다. 기본값은 4입니다. |
|  `format` |  (string) 비디오 형식으로, 경로나 io 오브젝트로 초기화할 때 필요합니다. |

#### 예시:

### numpy array를 비디오로 로그하기

```python
import numpy as np
import wandb

wandb.init()
# 축은 (time, channel, height, width)입니다.
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=4)})
```

## 메소드

### `encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/video.py#L130-L167)

```python
encode() -> None
```

| 클래스 변수 |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |