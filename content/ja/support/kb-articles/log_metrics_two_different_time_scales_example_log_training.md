---
title: 異なる 2 つの時間スケールでメトリクスをログできますか？
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

例えば、バッチごとにトレーニング精度、エポックごとに検証精度をログしたい場合を考えてみましょう。

はい、`batch` や `epoch` のようなインデックスもメトリクスと一緒にログできます。例えば `wandb.Run.log()({'train_accuracy': 0.9, 'batch': 200})` でトレーニング精度、`wandb.Run.log()({'val_accuracy': 0.8, 'epoch': 4})` で検証精度をそれぞれのタイミングで記録できます。UI ではグラフごとに任意の値を x 軸に設定できます。特定のインデックスをデフォルトの x 軸としたい場合は、[Run.define_metric()]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ja" >}}) を使ってください。下記の例では、次のように記述します。

```python
import wandb

with wandb.init() as run:
   # バッチを x 軸として定義
   run.define_metric("batch")
   # エポックを x 軸として定義
   run.define_metric("epoch")

   # トレーニング精度はバッチ単位で記録
   run.define_metric("train_accuracy", step_metric="batch")
   # 検証精度はエポック単位で記録
   run.define_metric("val_accuracy", step_metric="epoch")
```