---
title: Plotly
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Plotly
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/plotly.py >}}




## <kbd>class</kbd> `Plotly`
Plotly プロット用の W&B クラスです。

### <kbd>method</kbd> `Plotly.__init__`

```python
__init__(
    val: Union[ForwardRef('plotly.Figure'), ForwardRef('matplotlib.artist.Artist')]
)
```

Plotly オブジェクトを初期化します。


**引数:**
 
 - `val`:  Matplotlib または Plotly のフィギュア（図）を指定します。




---