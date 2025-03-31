---
menu:
  support:
    identifier: ja-support-kb-articles-log_metrics_batches_some_metrics_epochs
support:
- experiments
- metrics
title: What if I want to log some metrics on batches and some metrics only on epochs?
toc_hide: true
type: docs
url: /support/:filename
---

To log specific metrics in each batch and standardize plots, log the desired x-axis values alongside the metrics. In the custom plots, click edit and select a custom x-axis.

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```