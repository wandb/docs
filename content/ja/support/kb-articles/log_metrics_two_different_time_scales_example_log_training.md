---
title: 2 つの異なる時間スケールでメトリクスをログできますか？
menu:
  support:
    identifier: ja-support-kb-articles-log_metrics_two_different_time_scales_example_log_training
support:
- 実験
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

たとえば、バッチごとにトレーニング精度を、エポックごとに検証精度をログしたいとします。

はい、`batch` や `epoch` のようなインデックスをメトリクスと一緒にログしてください。1 つのステップでは `wandb.Run.log()({'train_accuracy': 0.9, 'batch': 200})` を使い、別のステップでは `wandb.Run.log()({'val_accuracy': 0.8, 'epoch': 4})` を使います。UI で、各チャートの x 軸にしたい値を設定します。特定のインデックスに対してデフォルトの x 軸を設定するには、[Run.define_metric()]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ja" >}}) を使用します。上の例では、次のコードを使用します:

```python
import wandb

with wandb.init() as run:
   run.define_metric("batch")
   run.define_metric("epoch")

   run.define_metric("train_accuracy", step_metric="batch")
   run.define_metric("val_accuracy", step_metric="epoch")
```