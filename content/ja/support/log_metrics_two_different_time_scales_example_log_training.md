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

例えば、バッチごとのトレーニング精度と、エポックごとの検証精度をログに記録したいとします。

はい、メトリクスと一緒に `batch` や `epoch` のような指標をログに記録します。あるステップで `wandb.log({'train_accuracy': 0.9, 'batch': 200})` を使用し、別のステップで `wandb.log({'val_accuracy': 0.8, 'epoch': 4})` を使用します。UI で、目的の値を各グラフの x 軸として設定します。特定の指標のデフォルトの x 軸を設定するには、[Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}}) を使用します。提供された例では、次のコードを使用します。

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```
