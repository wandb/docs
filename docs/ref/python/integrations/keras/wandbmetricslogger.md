# WandbMetricsLogger

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/metrics_logger.py#L23-L130' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&Bにシステムメトリクスを送信するロガー。

```python
WandbMetricsLogger(
    log_freq: Union[LogStrategy, int] = "epoch",
    initial_global_step: int = 0,
    *args,
    **kwargs
) -> None
```

`WandbMetricsLogger` は、コールバックメソッドが引数として受け取る `logs` 辞書を自動的にwandbにログします。

このコールバックはW&B runページに以下を自動的にログします:

* システム (CPU/GPU/TPU) メトリクス、
* `model.compile`で定義されたトレーニングおよびバリデーションのメトリクス、
* 学習率 (固定値または学習率スケジューラ)

#### メモ:

学習率スケジューラを使用して `model.fit` に `initial_epoch` を渡してトレーニングを再開する場合は、`WandbMetricsLogger`に `initial_global_step` を渡してください。`initial_global_step` は `step_size * initial_step` であり、ここで `step_size` は各エポックあたりのトレーニングステップ数です。`step_size` はトレーニングデータセットのカーディナリティとバッチサイズの積として計算できます。

| 引数 |  |
| :--- | :--- |
|  `log_freq` |  ("epoch", "batch" または int) "epoch"の場合、各エポックの終わりにメトリクスをログします。"batch"の場合、各バッチの終わりにメトリクスをログします。整数の場合、その数のバッチの終わりにメトリクスをログします。デフォルトは "epoch"。 |
|  `initial_global_step` |  (int) 一部の `initial_epoch` からトレーニングを再開し、学習率スケジューラを使用する場合に対応するために、この引数を使用して学習率を正しくログします。これは `step_size * initial_step` として計算できます。デフォルトは 0。 |

## メソッド

### `set_model`

```python
set_model(
    model
)
```

### `set_params`

```python
set_params(
    params
)
```