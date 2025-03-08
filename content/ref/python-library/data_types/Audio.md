---
title: Audio
object_type: data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/audio.py >}}




## <kbd>class</kbd> `Audio`
W&B class for audio clips. 



**Attributes:**
 
 - `data_or_path` (string or numpy array):  A path to an audio file  or a numpy array of audio data. 
 - `sample_rate` (int):  Sample rate, required when passing in raw  numpy array of audio data. 
 - `caption` (string):  Caption to display with audio. 

### <kbd>method</kbd> `Audio.__init__`

```python
__init__(data_or_path, sample_rate=None, caption=None)
```

Accept a path to an audio file or a numpy array of audio data. 




---









