# watch

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_watch.py#L20-L106' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

torchモデルにフックして、勾配とトポロジーを収集します。

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

任意のMLモデルを受け入れるように拡張する必要があります。

| 引数 |  |
| :--- | :--- |
|  `models` |  (torch.Module) フックするモデル、タプルも可能 |
|  `criterion` |  (torch.F) オプションの最適化対象の損失値 |
|  `log` |  (str) "gradients", "parameters", "all", またはNoneのいずれか |
|  `log_freq` |  (int) 勾配とパラメータをNバッチごとにログする |
|  `idx` |  (int) 複数のモデルに対してwandb.watchを呼び出す際に使用するインデックス |
|  `log_graph` |  (boolean) グラフトポロジーをログする |

| 戻り値 |  |
| :--- | :--- |
|  `wandb.Graph`: 最初のbackward pass後にポピュレートされるグラフオブジェクト |

| 例外 |  |
| :--- | :--- |
|  `ValueError` |  `wandb.init`の前に呼び出された場合、またはいずれかのモデルがtorch.nn.Moduleでない場合 |
