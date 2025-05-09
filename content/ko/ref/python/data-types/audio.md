---
title: Audio
menu:
  reference:
    identifier: ko-ref-python-data-types-audio
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/audio.py#L13-L157 >}}

오디오 클립을 위한 Wandb 클래스입니다.

```python
Audio(
    data_or_path, sample_rate=None, caption=None
)
```

| ARG |  |
| :--- | :--- |
|  `data_or_path` |  (string 또는 numpy array) 오디오 파일의 경로 또는 오디오 데이터의 numpy array입니다. |
|  `sample_rate` |  (int) 샘플 속도. 오디오 데이터의 raw numpy array를 전달할 때 필요합니다. |
|  `caption` |  (string) 오디오와 함께 표시할 캡션입니다. |

## 메소드 (method)

### `durations`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/audio.py#L115-L117)

```python
@classmethod
durations(
    audio_list
)
```

### `resolve_ref`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/audio.py#L131-L143)

```python
resolve_ref()
```

### `sample_rates`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/audio.py#L119-L121)

```python
@classmethod
sample_rates(
    audio_list
)
```