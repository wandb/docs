# ビデオ

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/video.py#L49-L238)

W&Bにログインするためのビデオをフォーマットします。

```python
Video(
 data_or_path: Union['np.ndarray', str, 'TextIO', 'BytesIO'],
 caption: Optional[str] = None,
 fps: int = 4,
 format: Optional[str] = None
)
```

| 引数 | 説明 |
| :--- | :--- |
| `data_or_path` | (numpy配列, 文字列, io) ビデオは、ファイルへのパスまたはioオブジェクトで初期化できます。フォーマットは "gif"、"mp4"、"webm"、または "ogg" である必要があります。フォーマットは、format引数で指定する必要があります。ビデオはnumpyテンソルで初期化することもできます。numpyテンソルは、4次元または5次元である必要があります。チャンネルは（時間, チャンネル, 高さ, 幅）または（バッチ, 時間, チャンネル, 高さ, 幅）である必要があります。|
| `caption` | (文字列) ビデオに関連するキャプション表示 |
| `fps` | (int) ビデオのフレームレート。デフォルトは4です。 |
| `format` | (文字列) ビデオのフォーマットで、パスまたはioオブジェクトで初期化する場合に必要です。 |
#### 例:

### numpy配列をビデオとしてログする

```python
import numpy as np
import wandb

wandb.init()
# 軸は (時間, チャンネル, 高さ, 幅)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=4)})
```

## メソッド

### `encode`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/video.py#L129-L166)

```python
encode() -> None
```
| クラス変数 | |

| :--- | :--- |

| `EXTS` | |