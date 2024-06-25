
# Video

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/video.py#L48-L239' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&Bに動画をログするためのフォーマット。

```python
Video(
    data_or_path: Union['np.ndarray', str, 'TextIO', 'BytesIO'],
    caption: Optional[str] = None,
    fps: int = 4,
    format: Optional[str] = None
)
```

| 引数 |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) 動画はファイルのパスやioオブジェクトで初期化できます。フォーマットは "gif", "mp4", "webm" または "ogg" でなければなりません。フォーマットはformat引数で指定する必要があります。動画はnumpyテンソルでも初期化できます。numpyテンソルは4次元または5次元でなければなりません。チャンネルは (time, channel, height, width) もしくは (batch, time, channel, height width) でなければなりません。 |
|  `caption` |  (string) 表示のために動画に関連付けられたキャプション |
|  `fps` |  (int) 動画のフレーム毎秒数。デフォルトは4です。 |
|  `format` |  (string) パスまたはioオブジェクトで初期化する場合に必要な動画のフォーマット。 |

#### 例:

### numpy arrayを動画としてログする

```python
import numpy as np
import wandb

wandb.init()
# 軸は (time, channel, height, width) です
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=4)})
```

## メソッド

### `encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/video.py#L130-L167)

```python
encode() -> None
```

| クラス変数 |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |