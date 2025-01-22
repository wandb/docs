---
title: Plotly
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/38b83df32bc652a763acb1345e687c88746bf647/wandb/sdk/data_types/plotly.py#L33-L82 >}}

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

[View source](https://www.github.com/wandb/wandb/tree/38b83df32bc652a763acb1345e687c88746bf647/wandb/sdk/data_types/plotly.py#L42-L50)

```python
@classmethod
make_plot_media(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```
