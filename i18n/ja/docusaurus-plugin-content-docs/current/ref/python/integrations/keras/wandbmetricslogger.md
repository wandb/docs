# WandbMetricsLogger

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/callbacks/metrics_logger.py#L23-L130)

W&Bにシステムメトリクスを送信するロガーです。

```python
WandbMetricsLogger(
 ログ頻度: Union[LogStrategy, int] = "エポック",
 初期グローバルステップ: int = 0,
 *args,
 **kwargs
) -> None
```

`WandbMetricsLogger`は、コールバックメソッドの引数として渡される`logs`辞書を自動的にwandbにログします。

このコールバックは、以下の内容をW&Bのrunページに自動的にログします。
* システム（CPU/GPU/TPU）メトリクス
* `model.compile`で定義されたトレーニングおよび検証メトリクス
* 学習率（固定値または学習率スケジューラの両方）
#### ノート:

`initial_epoch` を `model.fit` に渡してトレーニングを再開する場合、学習率スケジューラを使用している場合は、`WandbMetricsLogger` に `initial_global_step` を渡すことを確認してください。`initial_global_step` は、`step_size * initial_step` で、`step_size` はエポックごとのトレーニングステップ数です。`step_size`は、トレーニングデータセットのカーディナリティとバッチサイズの積として計算できます。

| 引数 |  |
| :--- | :--- |
| log_freq ("epoch", "batch", または int): "epoch" の場合、各エポックの終わりにメトリクスをログに記録します。"batch" の場合、各バッチの終わりにメトリクスをログに記録します。整数の場合、そのバッチ数の終わりにメトリクスをログに記録します。デフォルトは "epoch" です。initial_global_step (int): トレーニングをある `initial_epoch` から再開し、学習率スケジューラが使用されている場合、学習率を正しくログに記録するためにこの引数を使用します。これは `step_size * initial_step` として計算できます。デフォルトは0です。|

## メソッド

### `set_model`



```python
set_model(
 model
)```


### `set_params`







```python

set_params(

 params

)

```