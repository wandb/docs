---
title: メトリクスを異なる時間スケールでログすることはできますか？
menu:
  support:
    identifier: >-
      ja-support-kb-articles-log_metrics_two_different_time_scales_example_log_training
support:
  - experiments
  - metrics
toc_hide: true
type: docs
url: /ja/support/:filename
---
バッチごとのトレーニング精度とエポックごとの検証精度をログに記録したい場合を考えてみましょう。

はい、`batch` や `epoch` のようなインデックスをメトリクスと一緒にログに記録できます。`wandb.log({'train_accuracy': 0.9, 'batch': 200})` を使って、一つのステップで記録し、`wandb.log({'val_accuracy': 0.8, 'epoch': 4})` を別のステップで使用します。 UI では、各チャートの x 軸として目的の値を設定します。特定のインデックスに対してデフォルトの x 軸を設定するには、[Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}}) を使用します。提供された例に対しては、以下のコードを使用します。

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```