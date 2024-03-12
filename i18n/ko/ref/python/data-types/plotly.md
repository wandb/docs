
# Plotly

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/plotly.py#L33-L82' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


Plotly 그래프를 위한 Wandb 클래스.

```python
Plotly(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
)
```

| 인수 |  |
| :--- | :--- |
|  `val` |  matplotlib 또는 plotly 그림 |

## 메소드

### `make_plot_media`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/plotly.py#L42-L50)

```python
@classmethod
make_plot_media(
    val: Union['plotly.Figure', 'matplotlib.artist.Artist']
) -> Union[Image, 'Plotly']
```