---
title: Can I log metrics on two different time scales?
menu:
  support:
    identifier: ja-support-log_metrics_two_different_time_scales_example_log_training
tags:
- experiments
- metrics
toc_hide: true
type: docs
---

例えば、各バッチごとのトレーニング精度とエポックごとの検証精度をログに記録したいとします。

はい、`batch` や `epoch` などのインデックスをメトリクスと一緒にログに記録できます。`wandb.log({'train_accuracy': 0.9, 'batch': 200})` を一つのステップで使用し、`wandb.log({'val_accuracy': 0.8, 'epoch': 4})` を別のステップで使用します。UIでは、各チャートのx軸として希望する値を設定できます。特定のインデックスに対してデフォルトのx軸を設定するには、[Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}})を使用します。この例では、次のコードを使用してください。

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```