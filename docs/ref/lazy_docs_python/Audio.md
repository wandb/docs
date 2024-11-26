import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Audio

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/audio.py'/>




## <kbd>class</kbd> `Audio`
Wandb class for audio clips. 



**Args:**
 
 - `data_or_path`:  (string or numpy array) A path to an audio file  or a numpy array of audio data. 
 - `sample_rate`:  (int) Sample rate, required when passing in raw  numpy array of audio data. 
 - `caption`:  (string) Caption to display with audio. 

### <kbd>method</kbd> `Audio.__init__`

```python
__init__(data_or_path, sample_rate=None, caption=None)
```

Accept a path to an audio file or a numpy array of audio data. 




---

### <kbd>method</kbd> `Audio.bind_to_run`

```python
bind_to_run(run, key, step, id_=None, ignore_copy_err: Optional[bool] = None)
```





---

### <kbd>classmethod</kbd> `Audio.captions`

```python
captions(audio_list)
```





---

### <kbd>classmethod</kbd> `Audio.durations`

```python
durations(audio_list)
```





---

### <kbd>classmethod</kbd> `Audio.from_json`

```python
from_json(json_obj, source_artifact)
```





---

### <kbd>classmethod</kbd> `Audio.get_media_subdir`

```python
get_media_subdir()
```





---

### <kbd>method</kbd> `Audio.resolve_ref`

```python
resolve_ref()
```





---

### <kbd>classmethod</kbd> `Audio.sample_rates`

```python
sample_rates(audio_list)
```





---

### <kbd>classmethod</kbd> `Audio.seq_to_json`

```python
seq_to_json(seq, run, key, step)
```





---

### <kbd>method</kbd> `Audio.to_json`

```python
to_json(run)
```