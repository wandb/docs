---
title: Plotly
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/e35e545afd28aab70ee9e2a9dcc5ec7cfb95b1a1/wandb/sdk/data_types/plotly.py#L33-L82 >}}

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

[View source](https://www.github.com/wandb/wandb/tree/e35e545afd28aab70ee9e2a9dcc5ec7cfb95b1a1/wandb/sdk/data_types/plotly.py#L42-L50)

```python
@classmethod
make_plot_media(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```
