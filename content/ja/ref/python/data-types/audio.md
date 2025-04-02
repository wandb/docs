---
title: Audio
menu:
  reference:
    identifier: ja-ref-python-data-types-audio
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/audio.py#L13-L157 >}}

オーディオクリップ用の Wandb クラス。

```python
Audio(
    data_or_path, sample_rate=None, caption=None
)
```

| arg |  |
| :--- | :--- |
| `data_or_path` | (string または numpy 配列) オーディオファイルへのパス、またはオーディオ データの numpy 配列。 |
| `sample_rate` | (int) サンプルレート。生の numpy 配列のオーディオデータを渡す場合に必須。 |
| `caption` | (string) オーディオと一緒に表示するキャプション。 |

## メソッド

### `durations`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/audio.py#L115-L117)

```python
@classmethod
durations(
    audio_list
)
```

### `resolve_ref`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/audio.py#L131-L143)

```python
resolve_ref()
```

### `sample_rates`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/audio.py#L119-L121)

```python
@classmethod
sample_rates(
    audio_list
)
```