---
title: Plotly
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/sdk/data_types/plotly.py#L33-L95 >}}

W&B class for Plotly plots.

## Methods

### `make_plot_media`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/sdk/data_types/plotly.py#L38-L50)

```python
@classmethod
make_plot_media(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```

Create a Plotly object from a Plotly figure or a matplotlib artist.

<!-- lazydoc-ignore-classmethod: internal -->
