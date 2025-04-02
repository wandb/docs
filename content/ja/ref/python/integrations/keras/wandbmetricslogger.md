---
title: WandbMetricsLogger
menu:
  reference:
    identifier: ja-ref-python-integrations-keras-wandbmetricslogger
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/metrics_logger.py#L16-L129 >}}

システムメトリクスを W&B に送信するロガー。

```python
WandbMetricsLogger(
    log_freq: Union[LogStrategy, int] = "epoch",
    initial_global_step: int = 0,
    *args,
    **kwargs
) -> None
```

`WandbMetricsLogger` は、コールバックメソッドが wandb に引数として取る `logs` 辞書を自動的に ログ 記録します。

このコールバックは、以下のものを自動的に W&B の run ページにログ記録します。

* システム (CPU/GPU/TPU) メトリクス
* `model.compile` で定義されたトレーニングおよび検証メトリクス
* 学習率（固定値と学習率スケジューラの両方）

#### 注:

`initial_epoch` を `model.fit` に渡してトレーニングを再開し、学習率スケジューラを使用している場合は、`initial_global_step` を `WandbMetricsLogger` に渡してください。`initial_global_step` は `step_size * initial_step` です。`step_size` は、エポックごとのトレーニングステップ数です。`step_size` は、トレーニングデータセットのカーディナリティとバッチサイズの積として計算できます。

| 引数 |  |
| :--- | :--- |
|  `log_freq` |  ("epoch"、"batch"、または int) "epoch" の場合、各エポックの最後にメトリクスをログ記録します。"batch" の場合、各バッチの最後にメトリクスをログ記録します。整数である場合、その数のバッチの最後にメトリクスをログ記録します。デフォルトは "epoch" です。 |
|  `initial_global_step` |  (int) `initial_epoch` からトレーニングを再開し、学習率スケジューラを使用している場合は、この引数を使用して学習率を正しくログ記録します。これは `step_size * initial_step` として計算できます。デフォルトは 0 です。 |

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
