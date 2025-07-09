---
title: Audio
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/audio.py#L17-L172 >}}

Wandb class for audio clips.

| Args |  |
| :--- | :--- |
|  `data_or_path` |  (string or numpy array) A path to an audio file or a numpy array of audio data. |
|  `sample_rate` |  (int) Sample rate, required when passing in raw numpy array of audio data. |
|  `caption` |  (string) Caption to display with audio. |

## Methods

### `durations`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/audio.py#L130-L132)

```python
@classmethod
durations(
    audio_list
)
```

### `resolve_ref`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/audio.py#L146-L158)

```python
resolve_ref()
```

### `sample_rates`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/audio.py#L134-L136)

```python
@classmethod
sample_rates(
    audio_list
)
```
