# Plotly



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/sdk/data_types/plotly.py#L32-L82)



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



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/sdk/data_types/plotly.py#L42-L50)

```python
@classmethod
make_plot_media(
 val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```






