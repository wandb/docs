---
title: ビデオ
menu:
  reference:
    identifier: ja-ref-python-data-types-video
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/video.py#L49-L251 >}}

W&B にログするためのビデオをフォーマットします。

```python
Video(
    data_or_path: Union['np.ndarray', str, 'TextIO', 'BytesIO'],
    caption: Optional[str] = None,
    fps: Optional[int] = None,
    format: Optional[str] = None
)
```

| 引数 |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) ビデオはファイルへのパスまたは io オブジェクトで初期化できます。フォーマットは "gif", "mp4", "webm", "ogg" のいずれかでなければなりません。フォーマットは format 引数で指定する必要があります。ビデオは numpy テンソルでも初期化できます。numpy テンソルは 4次元または 5次元でなければなりません。チャンネルは (time, channel, height, width) または (batch, time, channel, height, width) であるべきです。 |
|  `caption` |  (string) ビデオに関連付けられたキャプション（表示用） |
|  `fps` |  (int) 生のビデオフレームをエンコードする際のフレームレート。デフォルト値は 4 です。このパラメータは data_or_path が string または bytes の場合には影響しません。 |
|  `format` |  (string) ビデオのフォーマット。パスまたは io オブジェクトで初期化する場合に必要です。 |

#### 例:

### numpy 配列をビデオとしてログする

```python
import numpy as np
import wandb

run = wandb.init()
# 軸は (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

## メソッド

### `encode`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/video.py#L140-L179)

```python
encode(
    fps: int = 4
) -> None
```

| クラス変数 |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |