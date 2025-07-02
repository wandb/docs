---
title: Plotly
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/plotly.py#L33-L82 >}}

Wandb class for plotly plots.

| Args |  |
| :--- | :--- |
|  `val` |  matplotlib or plotly figure |

## Methods

### `make_plot_media`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/plotly.py#L42-L50)

```python
@classmethod
make_plot_media(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```
