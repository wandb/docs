
# WandbMetricsLogger

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/metrics_logger.py#L23-L130' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&B にシステムメトリクスを送信するロガー。

```python
WandbMetricsLogger(
    log_freq: Union[LogStrategy, int] = "epoch",
    initial_global_step: int = 0,
    *args,
    **kwargs
) -> None
```

`WandbMetricsLogger` は、コールバックメソッドが引数として取る `logs` 辞書を自動的に wandb にログします。

このコールバックは、W&B の run ページに次の内容を自動的にログします:

* システム (CPU/GPU/TPU) メトリクス
* `model.compile` で定義されたトレーニングおよび検証メトリクス
* 学習率（固定値または学習率スケジューラの場合）

#### 注意事項:

`initial_epoch` を `model.fit` に渡してトレーニングを再開し、学習率スケジューラを使用している場合は、`WandbMetricsLogger` に `initial_global_step` を渡すようにしてください。`initial_global_step` は `step_size * initial_step` であり、`step_size` はエポックごとのトレーニングステップ数です。`step_size` はトレーニングデータセットのカーディナリティとバッチサイズの積として計算できます。

| 引数 |  |
| :--- | :--- |
|  `log_freq` |  ("epoch", "batch", または int) "epoch" の場合、各エポックの終了時にメトリクスをログします。 "batch" の場合、各バッチの終了時にメトリクスをログします。 整数の場合、その多くのバッチの終了時にメトリクスをログします。 デフォルトは "epoch" です。 |
|  `initial_global_step` |  (int) トレーニングを `initial_epoch` から再開しており、学習率スケジューラを使っている場合、この引数を使って学習率を正しくログします。 これは `step_size * initial_step` として計算できます。 デフォルトは 0 です。 |

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