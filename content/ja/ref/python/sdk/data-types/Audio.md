---
title: オーディオ
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Audio
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/audio.py >}}




## <kbd>クラス</kbd> `Audio`
W&B の音声クリップ用クラスです。 

### <kbd>メソッド</kbd> `Audio.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, list, ForwardRef('np.ndarray')],
    sample_rate: Optional[int] = None,
    caption: Optional[str] = None
)
```

音声ファイルへのパス、または音声データの NumPy 配列を受け取ります。 



**Args:**
 
 - `data_or_path`:  音声ファイルへのパス、または音声データの NumPy 配列。 
 - `sample_rate`:  サンプル レート。音声データの生の NumPy 配列を渡す場合は必須です。 
 - `caption`:  音声と一緒に表示するキャプション。 




---



### <kbd>クラス メソッド</kbd> `Audio.durations`

```python
durations(audio_list)
```

音声ファイルの再生時間を計算します。 

---



### <kbd>クラス メソッド</kbd> `Audio.sample_rates`

```python
sample_rates(audio_list)
```

音声ファイルのサンプル レートを取得します。 

---