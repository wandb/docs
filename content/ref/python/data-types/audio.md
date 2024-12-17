---
title: Audio
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/audio.py#L13-L157" >}}

Wandb class for audio clips.

```python
Audio(
    data_or_path, sample_rate=None, caption=None
)
```

| Args |  |
| :--- | :--- |
|  `data_or_path` |  (string or numpy array) A path to an audio file or a numpy array of audio data. |
|  `sample_rate` |  (int) Sample rate, required when passing in raw numpy array of audio data. |
|  `caption` |  (string) Caption to display with audio. |

## Methods

### `durations`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/audio.py#L115-L117)

```python
@classmethod
durations(
    audio_list
)
```

### `resolve_ref`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/audio.py#L131-L143)

```python
resolve_ref()
```

### `sample_rates`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/audio.py#L119-L121)

```python
@classmethod
sample_rates(
    audio_list
)
```
