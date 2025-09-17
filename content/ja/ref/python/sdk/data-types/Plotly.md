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
Plotly のプロット用の W&B クラス。 

### <kbd>method</kbd> `Plotly.__init__`

```python
__init__(
    val: Union[ForwardRef('plotly.Figure'), ForwardRef('matplotlib.artist.Artist')]
)
```

Plotly のオブジェクトを初期化します。 



**Args:**
 
 - `val`:  Matplotlib または Plotly の Figure。 




---