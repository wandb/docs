# Plotly

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/plotly.py#L32-L81)

Wandbでplotlyプロットのためのクラス。

```python
Plotly(
 val: Union['plotly.Figure', 'matplotlib.artist.Artist']
)
```

| 引数 | |
| :--- | :--- |
| `val` | matplotlibまたはplotlyの図表 |

## メソッド

### `make_plot_media`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/plotly.py#L41-L49)

```python
@classmethod
make_plot_media(
 val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```