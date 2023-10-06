# watch

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_watch.py#L20-L106)

トーチモデルにフックして、勾配とトポロジーを収集します。

```python
watch(
 models,
 criterion=None,
 log: Optional[Literal['gradients', 'parameters', 'all']] = "gradients",
 log_freq: int = 1000,
 idx: Optional[int] = None,
 log_graph: bool = (False)
)
```

任意のMLモデルを受け入れるように拡張すべきです。

| 引数 | 説明 |
| :--- | :--- |
| `models` | (torch.Module) フックするモデル。タプルであってもよい |
| `criterion` | (torch.F) 最適化されるオプションの損失値 |
| `log` | (str) "gradients"、"parameters"、"all"、または None のいずれか |
| `log_freq` | (int) Nバッチごとに勾配とパラメータをログする |
| `idx` | (int) 複数のモデルでwandb.watchを呼び出すときに使用されるインデックス |
| `log_graph` | (boolean) グラフトポロジーをログする |
| 返り値 | |

| :--- | :--- |

| `wandb.Graph` | 最初のbackwardパスの後にデータが入るグラフオブジェクト |



| 例外 | |

| :--- | :--- |

| `ValueError` | `wandb.init`が呼ばれる前に呼び出された場合、またはどのモデルもtorch.nn.Moduleでない場合。 |