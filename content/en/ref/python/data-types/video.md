---
title: Video
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.0/wandb/sdk/data_types/video.py#L65-L268 >}}

A class for logging videos to W&B.

```python
Video(
    data_or_path: Union[str, pathlib.Path, 'np.ndarray', 'TextIO', 'BytesIO'],
    caption: Optional[str] = None,
    fps: Optional[int] = None,
    format: Optional[Literal['gif', 'mp4', 'webm', 'ogg']] = None
)
```

| Args |  |
| :--- | :--- |
|  `data_or_path` |  Video can be initialized with a path to a file or an io object. Video can be initialized with a numpy tensor. The numpy tensor must be either 4 dimensional or 5 dimensional. The dimensions should be (number of frames, channel, height, width) or (batch, number of frames, channel, height, width) The format parameter must be specified with the format argument when initializing with a numpy array or io object. |
|  `caption` |  Caption associated with the video for display. |
|  `fps` |  The frame rate to use when encoding raw video frames. Default value is 4. This parameter has no effect when data_or_path is a string, or bytes. |
|  `format` |  Format of video, necessary if initializing with a numpy array or io object. This parameter will be used to determine the format to use when encoding the video data. Accepted values are "gif", "mp4", "webm", or "ogg". If no value is provided, the default format will be "gif". |

## Methods

### `encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.0/wandb/sdk/data_types/video.py#L176-L198)

```python
encode(
    fps: int = 4
) -> None
```

| Class Variables |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |
