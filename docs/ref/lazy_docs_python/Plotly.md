import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Plotly

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/plotly.py'/>




## <kbd>class</kbd> `Plotly`
Wandb class for plotly plots. 



**Args:**
 
 - `val`:  matplotlib or plotly figure 

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