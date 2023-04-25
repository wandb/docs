# Plotly



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/plotly.py#L32-L81)



Wandb class for plotly plots.

```python
Plotly(
 val: Union['plotly.Figure', 'matplotlib.artist.Artist']
)
```





| Arguments | |
| :--- | :--- |
| `val` | matplotlib or plotly figure |



## Methods

### `make_plot_media`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/plotly.py#L41-L49)

```python
@classmethod
make_plot_media(
 val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```






