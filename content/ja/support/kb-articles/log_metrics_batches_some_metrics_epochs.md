---
title: What if I want to log some metrics on batches and some metrics only on epochs?
menu:
  support:
    identifier: ja-support-kb-articles-log_metrics_batches_some_metrics_epochs
support:
- experiments
- metrics
toc_hide: true
type: docs
url: /support/:filename
---

各バッチで特定のメトリクスをログに記録し、プロットを標準化するには、目的のX軸の 値 と共にメトリクスをログに記録します。カスタムプロットで、編集をクリックし、カスタムX軸を選択します。

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```
