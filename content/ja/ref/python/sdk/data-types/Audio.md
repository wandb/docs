---
title: オーディオ
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Audio
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/audio.py >}}




## <kbd>class</kbd> `Audio`
W&B のオーディオクリップ用クラスです。

### <kbd>method</kbd> `Audio.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, list, ForwardRef('np.ndarray')],
    sample_rate: Optional[int] = None,
    caption: Optional[str] = None
)
```

オーディオファイルへのパス、またはオーディオ データが含まれる numpy 配列を受け取ります。

**Args:**
 
 - `data_or_path`:  オーディオファイルへのパス、またはオーディオ データが含まれる NumPy 配列。 
 - `sample_rate`:  生の NumPy 配列のオーディオ データを渡す場合に必要なサンプルレート。 
 - `caption`:  オーディオと一緒に表示するキャプション。 




---



### <kbd>classmethod</kbd> `Audio.durations`

```python
durations(audio_list)
```

オーディオファイルの長さ（再生時間）を計算します。

---



### <kbd>classmethod</kbd> `Audio.sample_rates`

```python
sample_rates(audio_list)
```

オーディオファイルのサンプルレートを取得します。

---