---
title: ビデオ
object_type: python_sdk_data_type
data_type_classification: class
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
 
 - `data_or_path`:  動画はファイルのパスまたは io オブジェクトで初期化できます。numpy テンソルでも初期化可能です。numpy テンソルの場合、4 次元または 5 次元の配列である必要があります。次元の構成は (フレーム数, チャンネル, 高さ, 幅) または (バッチ, フレーム数, チャンネル, 高さ, 幅) です。numpy 配列や io オブジェクトで初期化する場合、format パラメータを指定する必要があります。
 - `caption`:  動画と一緒に表示するキャプション。
 - `fps`:  生フレームをエンコードする際のフレームレート。デフォルト値は 4 です。このパラメータは、data_or_path が文字列またはバイト列の場合は影響しません。
 - `format`:  動画のフォーマット。numpy 配列や io オブジェクトで初期化する場合に指定が必要です。このパラメータで動画データのエンコードフォーマットが決まります。使用できる値は "gif"、"mp4"、"webm"、"ogg" です。値を指定しない場合はデフォルトで "gif" になります。



**例:**
 numpy 配列を動画としてログする

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 次元は (フレーム数, チャンネル, 高さ, 幅)
    frames = np.random.randint(
         low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8
    )
    run.log({"video": wandb.Video(frames, format="mp4", fps=4)})
``` 




---