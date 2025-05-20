---
title: Video
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/video.py >}}




## <kbd>class</kbd> `Video`
A class for logging videos to W&B. 



**Args:**
 
 - `data_or_path`:  Video can be initialized with a path to a file or an  io object. The format must be "gif", "mp4", "webm" or "ogg".  The format must be specified with the format argument.  Video can be initialized with a numpy tensor.  The numpy tensor must be either 4 dimensional or 5 dimensional.  Channels should be (time, channel, height, width) or  (batch, time, channel, height width) 
 - `caption`:  Caption associated with the video for display. 
 - `fps`:  The frame rate to use when encoding raw video frames.  Default value is 4. This parameter has no effect when  data_or_path is a string, or bytes. 
 - `format`:  Format of video, necessary if initializing with path or io  object. 



**Examples:**
 Log a numpy array as a video 

```python
import numpy as np
import wandb

run = wandb.init()
# axes are (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
``` 

### <kbd>method</kbd> `Video.__init__`

```python
__init__(
    data_or_path: Union[ForwardRef('np.ndarray'), str, ForwardRef('TextIO'), ForwardRef('BytesIO')],
    caption: Optional[str] = None,
    fps: Optional[int] = None,
    format: Optional[Literal['gif', 'mp4', 'webm', 'ogg']] = None
)
```

Initialize a W&B Video object. 



**Args:**
  data_or_path:  Video can be initialized with a path to a file or an io object.  Video can be initialized with a numpy tensor.  The numpy tensor must be either 4 dimensional or 5 dimensional.  The dimensions should be (number of frames, channel, height, width) or  (batch, number of frames, channel, height, width)  The format parameter must be specified with the format argument  when initializing with a numpy array  or io object. 
 - `caption`:  Caption associated with the video for display. fps:  The frame rate to use when encoding raw video frames.  Default value is 4.  This parameter has no effect when data_or_path is a string, or bytes. format:  Format of video, necessary if initializing with a numpy array  or io object. This parameter will be used to determine the format  to use when encoding the video data. Accepted values are "gif",  "mp4", "webm", or "ogg". 



**Examples:**
 Log a numpy array as a video ```python
import numpy as np
import wandb

with wandb.init() as run:
    # axes are (number of frames, channel, height, width)
    frames = np.random.randint(
         low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8
    )
    run.log({"video": wandb.Video(frames, format="mp4", fps=4)})
``` 




---




