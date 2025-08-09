---
title: Plotly
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-Plotly
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/plotly.py >}}




## <kbd>class</kbd> `Plotly`
W&B의 Plotly 플롯을 위한 클래스입니다.

### <kbd>method</kbd> `Plotly.__init__`

```python
__init__(
    val: Union[ForwardRef('plotly.Figure'), ForwardRef('matplotlib.artist.Artist')]
)
```

Plotly 오브젝트를 초기화합니다.

**ARG:**
 
 - `val`:  Matplotlib 또는 Plotly figure를 입력합니다.




---