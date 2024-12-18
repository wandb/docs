---
title: Video
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/video.py#L49-L249" >}}


Format a video for logging to W&B.

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
|  `data_or_path` |  (numpy array, string, io) Video can be initialized with a path to a file or an io object. The format must be "gif", "mp4", "webm" or "ogg". The format must be specified with the format argument. Video can be initialized with a numpy tensor. The numpy tensor must be either 4 dimensional or 5 dimensional. Channels should be (time, channel, height, width) or (batch, time, channel, height width) |
|  `caption` |  (string) caption associated with the video for display |
|  `fps` |  (int) The frame rate to use when encoding raw video frames. Default value is 4. This parameter has no effect when data_or_path is a string, or bytes. |
|  `format` |  (string) format of video, necessary if initializing with path or io object. |

#### Examples:

### Log a numpy array as a video

<!--yeadoc-test:log-video-numpy-->


```python
import numpy as np
import wandb

wandb.init()
# axes are (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=4)})
```

## Methods

### `encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/video.py#L140-L177)

```python
encode(
    fps: int = 4
) -> None
```

| Class Variables |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |
