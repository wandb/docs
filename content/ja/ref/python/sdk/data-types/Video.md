---
title: ビデオ
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Video
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/video.py >}}




## <kbd>class</kbd> `Video`
W&B に動画をログするためのクラスです。

### <kbd>method</kbd> `Video.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, ForwardRef('np.ndarray'), ForwardRef('TextIO'), ForwardRef('BytesIO')],
    caption: Optional[str] = None,
    fps: Optional[int] = None,
    format: Optional[Literal['gif', 'mp4', 'webm', 'ogg']] = None
)
```

W&B Video オブジェクトを初期化します。


**引数:**
 
 - `data_or_path`: ファイルへのパスや io オブジェクトで Video を初期化できます。numpy テンソルでも初期化できます。numpy テンソルの場合、4次元または 5次元である必要があります。次元は (フレーム数, チャンネル, 高さ, 幅) もしくは (バッチ, フレーム数, チャンネル, 高さ, 幅) でなければなりません。numpy 配列や io オブジェクトで初期化する場合は、format パラメータを指定する必要があります。
 - `caption`: 動画に関連付けられるキャプション（表示用）。
 - `fps`: 生動画フレームをエンコードする際に使用するフレームレート。デフォルト値は 4 です。このパラメータは data_or_path が文字列またはバイト列の場合には影響しません。
 - `format`: 動画のフォーマット。numpy 配列や io オブジェクトで初期化する場合に必要です。このパラメータは動画データをエンコードする際の形式を決定します。利用可能な値は "gif"、"mp4"、"webm"、"ogg" です。値が指定されていない場合、デフォルトは "gif" になります。


**使用例:**
 numpy 配列を動画としてログする

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 軸は (フレーム数, チャンネル, 高さ, 幅) です
    frames = np.random.randint(
         low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8
    )
    run.log({"video": wandb.Video(frames, format="mp4", fps=4)})
``` 




---