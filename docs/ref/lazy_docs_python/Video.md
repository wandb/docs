import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Video

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/video.py'/>




## <kbd>class</kbd> `Video`
Format a video for logging to W&B. 



**Args:**
 
 - `data_or_path`:  (numpy array, string, io)  Video can be initialized with a path to a file or an io object.  The format must be "gif", "mp4", "webm" or "ogg".  The format must be specified with the format argument.  Video can be initialized with a numpy tensor.  The numpy tensor must be either 4 dimensional or 5 dimensional.  Channels should be (time, channel, height, width) or  (batch, time, channel, height width) 
 - `caption`:  (string) caption associated with the video for display 
 - `fps`:  (int)  The frame rate to use when encoding raw video frames. Default value is 4.  This parameter has no effect when data_or_path is a string, or bytes. 
 - `format`:  (string) format of video, necessary if initializing with path or io object. 





### <kbd>method</kbd> `Video.__init__`

```python
__init__(
    data_or_path: Union[ForwardRef('np.ndarray'), str, ForwardRef('TextIO'), ForwardRef('BytesIO')],
    caption: Optional[str] = None,
    fps: Optional[int] = None,
    format: Optional[str] = None
)
```








---

### <kbd>method</kbd> `Video.encode`

```python
encode(fps: int = 4) → None
```





---

### <kbd>classmethod</kbd> `Video.get_media_subdir`

```python
get_media_subdir() → str
```





---

### <kbd>classmethod</kbd> `Video.seq_to_json`

```python
seq_to_json(
    seq: Sequence[ForwardRef('BatchableMedia')],
    run: 'LocalRun',
    key: str,
    step: Union[int, str]
) → dict
```





---

### <kbd>method</kbd> `Video.to_json`

```python
to_json(
    run_or_artifact: Union[ForwardRef('LocalRun'), ForwardRef('Artifact')]
) → dict
```