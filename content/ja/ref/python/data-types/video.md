---
title: Video
menu:
  reference:
    identifier: ja-ref-python-data-types-video
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/video.py#L49-L251 >}}

W&B に ログ を記録するためのビデオをフォーマットします。

```python
Video(
    data_or_path: Union['np.ndarray', str, 'TextIO', 'BytesIO'],
    caption: Optional[str] = None,
    fps: Optional[int] = None,
    format: Optional[str] = None
)
```

| Args |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) Video は、ファイルへのパスまたは io オブジェクトで初期化できます。フォーマットは "gif"、"mp4"、"webm" または "ogg" である必要があります。フォーマットは format 引数で指定する必要があります。Video は numpy テンソルで初期化できます。numpy テンソルは、4 次元または 5 次元である必要があります。チャンネルは (time, channel, height, width) または (batch, time, channel, height width) にする必要があります。 |
|  `caption` |  (string) 表示するビデオに関連付けられたキャプション |
|  `fps` |  (int) 生のビデオフレームをエンコードする際に使用するフレームレート。デフォルト値は 4 です。この パラメータ は、data_or_path が string または bytes の場合、効果はありません。 |
|  `format` |  (string) ビデオのフォーマット。パスまたは io オブジェクトで初期化する場合に必要です。 |

#### 例:

### numpy array をビデオとして ログ 記録する

```python
import numpy as np
import wandb

run = wandb.init()
# axes are (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

## メソッド

### `encode`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/video.py#L140-L179)

```python
encode(
    fps: int = 4
) -> None
```

| Class Variables |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |
