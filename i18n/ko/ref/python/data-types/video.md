
# 비디오

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/video.py#L48-L237' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

W&B에서 로깅하기 위한 비디오 포맷.

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
|  `data_or_path` |  (numpy array, string, io) 비디오는 파일 경로나 io 객체로 초기화할 수 있습니다. 포맷은 "gif", "mp4", "webm", "ogg"이어야 합니다. 포맷은 format 인수로 명시해야 합니다. 비디오는 numpy 텐서로 초기화할 수 있습니다. numpy 텐서는 4차원이나 5차원이어야 합니다. 채널은 (시간, 채널, 높이, 너비) 또는 (배치, 시간, 채널, 높이, 너비)이어야 합니다. |
|  `caption` |  (string) 비디오와 연관된 캡션을 표시합니다. |
|  `fps` |  (int) 비디오의 초당 프레임 수입니다. 기본값은 4입니다. |
|  `format` |  (string) 비디오의 포맷입니다, 경로나 io 객체로 초기화할 경우 필요합니다. |

#### 예시:

### numpy 배열을 비디오로 로그하기

```python
import numpy as np
import wandb

wandb.init()
# 축은 (시간, 채널, 높이, 너비)입니다.
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=4)})
```

## 메서드

### `encode`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/video.py#L128-L165)

```python
encode() -> None
```

| 클래스 변수 |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |