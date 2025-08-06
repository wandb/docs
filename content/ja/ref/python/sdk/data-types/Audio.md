---
title: オーディオ
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/audio.py >}}




## <kbd>class</kbd> `Audio`
W&B の音声クリップ用クラスです。

### <kbd>method</kbd> `Audio.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, list, ForwardRef('np.ndarray')],
    sample_rate: Optional[int] = None,
    caption: Optional[str] = None
)
```

音声ファイルへのパス、または音声データの numpy 配列を受け付けます。

**Args:**
 
 - `data_or_path`:  音声ファイルへのパス、もしくは音声データの NumPy 配列。
 - `sample_rate`:  生の NumPy 配列を渡す場合に必須となるサンプリングレート。
 - `caption`:  音声と一緒に表示するキャプション。




---



### <kbd>classmethod</kbd> `Audio.durations`

```python
durations(audio_list)
```

音声ファイルの長さ（再生時間）を計算します。

---



### <kbd>classmethod</kbd> `Audio.sample_rates`

```python
sample_rates(audio_list)
```

音声ファイルのサンプリングレートを取得します。

---