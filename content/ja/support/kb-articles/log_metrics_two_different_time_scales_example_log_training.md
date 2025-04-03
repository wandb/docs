---
title: Can I log metrics on two different time scales?
menu:
  support:
    identifier: ja-support-kb-articles-log_metrics_two_different_time_scales_example_log_training
support:
- experiments
- metrics
toc_hide: true
type: docs
url: /support/:filename
---

例えば、バッチごとのトレーニング精度と、エポックごとの検証精度をログに記録したいとします。

はい、`batch` や `epoch` のようなインデックスを メトリクス と共にログに記録してください。`wandb.log({'train_accuracy': 0.9, 'batch': 200})` をあるステップで使用し、`wandb.log({'val_accuracy': 0.8, 'epoch': 4})` を別のステップで使用します。UIで、目的の 値 を各グラフのx軸として設定します。特定のインデックスのデフォルトのx軸を設定するには、[Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}}) を使用してください。提供された例では、次の コード を使用します。

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```
