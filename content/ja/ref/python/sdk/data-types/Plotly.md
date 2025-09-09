---
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Plotly
object_type: python_sdk_data_type
title: Plotly
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/plotly.py >}}




## <kbd>class</kbd> `Plotly`
W&B class for Plotly plots. 

### <kbd>method</kbd> `Plotly.__init__`

```python
__init__(
    val: Union[ForwardRef('plotly.Figure'), ForwardRef('matplotlib.artist.Artist')]
)
```

Initialize a Plotly object. 



**Args:**
 
 - `val`:  Matplotlib or Plotly figure. 




---