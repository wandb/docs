---
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Audio
object_type: python_sdk_data_type
title: Audio
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/audio.py >}}




## <kbd>class</kbd> `Audio`
W&B class for audio clips. 

### <kbd>method</kbd> `Audio.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, list, ForwardRef('np.ndarray')],
    sample_rate: Optional[int] = None,
    caption: Optional[str] = None
)
```

Accept a path to an audio file or a numpy array of audio data. 



**Args:**
 
 - `data_or_path`:  A path to an audio file or a NumPy array of audio data. 
 - `sample_rate`:  Sample rate, required when passing in raw NumPy array of audio data. 
 - `caption`:  Caption to display with audio. 




---



### <kbd>classmethod</kbd> `Audio.durations`

```python
durations(audio_list)
```

Calculate the duration of the audio files. 

---



### <kbd>classmethod</kbd> `Audio.sample_rates`

```python
sample_rates(audio_list)
```

Get sample rates of the audio files. 

---