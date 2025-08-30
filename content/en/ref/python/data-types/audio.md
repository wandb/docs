---
title: Audio
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.2/wandb/sdk/data_types/audio.py#L17-L200 >}}

W&B class for audio clips.

## Methods

### `durations`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.2/wandb/sdk/data_types/audio.py#L148-L151)

```python
@classmethod
durations(
    audio_list
)
```

Calculate the duration of the audio files.

### `resolve_ref`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.2/wandb/sdk/data_types/audio.py#L170-L186)

```python
resolve_ref()
```

Resolve the reference to the actual file path.

<!-- lazydoc-ignore: internal -->


### `sample_rates`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.2/wandb/sdk/data_types/audio.py#L153-L156)

```python
@classmethod
sample_rates(
    audio_list
)
```

Get sample rates of the audio files.
