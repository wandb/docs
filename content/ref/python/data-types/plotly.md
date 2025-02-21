---
title: Plotly
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/2678738e59629208ad4770e3d36300a272147c05/wandb/sdk/data_types/plotly.py#L33-L82 >}}

Wandb class for plotly plots.

```python
Plotly(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
)
```

| Args |  |
| :--- | :--- |
|  `val` |  matplotlib or plotly figure |

## Methods

### `make_plot_media`

[View source](https://www.github.com/wandb/wandb/tree/2678738e59629208ad4770e3d36300a272147c05/wandb/sdk/data_types/plotly.py#L42-L50)

```python
@classmethod
make_plot_media(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```
