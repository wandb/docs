---
title: What if I want to log some metrics on batches and some metrics only on epochs?
menu:
  support:
    identifier: ja-support-log_metrics_batches_some_metrics_epochs
tags:
- experiments
- metrics
toc_hide: true
type: docs
---

各バッチで特定のメトリクスをログし、プロットを標準化するために、メトリクスと一緒に目的の x 軸の値をログします。カスタムプロットでは、編集をクリックしてカスタム x 軸を選択します。

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```
