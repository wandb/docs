---
title: Plotly
object_type: data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/plotly.py >}}




## <kbd>class</kbd> `Plotly`
W&B class for Plotly plots. 



**Args:**
 
 - `val`:  Matplotlib or Plotly figure. 

### <kbd>method</kbd> `Plotly.__init__`

```python
__init__(
    val: Union[ForwardRef('plotly.Figure'), ForwardRef('matplotlib.artist.Artist')]
)
```








---

### <kbd>classmethod</kbd> `Plotly.get_media_subdir`

```python
get_media_subdir() → str
```





---

### <kbd>classmethod</kbd> `Plotly.make_plot_media`

```python
make_plot_media(
    val: Union[ForwardRef('plotly.Figure'), ForwardRef('matplotlib.artist.Artist')]
) → Union[wandb.sdk.data_types.image.Image, ForwardRef('Plotly')]
```





---

### <kbd>method</kbd> `Plotly.to_json`

```python
to_json(
    run_or_artifact: Union[ForwardRef('LocalRun'), ForwardRef('Artifact')]
) → dict
```





