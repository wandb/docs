---
title: Plotly
menu:
  reference:
    identifier: ko-ref-python-data-types-plotly
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/plotly.py#L33-L82 >}}

plotly plot을 위한 Wandb 클래스입니다.

```python
Plotly(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
)
```

| ARG |  |
| :--- | :--- |
|  `val` |  matplotlib 또는 plotly figure |

## 메소드

### `make_plot_media`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/plotly.py#L42-L50)

```python
@classmethod
make_plot_media(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```
