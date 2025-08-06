---
title: 2 つの異なる時間スケールでメトリクスをログできますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
- メトリクス
---

例えば、各バッチごとにトレーニング精度を、各エポックごとに検証精度をログしたい場合。

はい、`batch` や `epoch` のようなログインデックスをメトリクスと一緒に記録できます。1 回のステップで `wandb.Run.log()({'train_accuracy': 0.9, 'batch': 200})`、別のステップで `wandb.Run.log()({'val_accuracy': 0.8, 'epoch': 4})` のように使います。UI では、各チャートごとに希望の値を x 軸に設定できます。特定のインデックスをデフォルトの x 軸にしたい場合は、[Run.define_metric()]({{< relref "/ref/python/sdk/classes/run#define_metric" >}}) を利用してください。今回の例の場合、次のコードを使用します。

```python
import wandb

with wandb.init() as run:
   # バッチごとのメトリクスを定義
   run.define_metric("batch")
   # エポックごとのメトリクスを定義
   run.define_metric("epoch")

   # トレーニング精度は batch を x 軸（step metric）に
   run.define_metric("train_accuracy", step_metric="batch")
   # 検証精度は epoch を x 軸（step metric）に
   run.define_metric("val_accuracy", step_metric="epoch")
```