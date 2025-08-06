---
menu:
  support:
    identifier: ko-support-kb-articles-log_metrics_two_different_time_scales_example_log_training
support:
- experiments
- metrics
title: Can I log metrics on two different time scales?
toc_hide: true
type: docs
url: /support/:filename
---

For example, I want to log training accuracy per batch and validation accuracy per epoch.

Yes, log indices like `batch` and `epoch` alongside your metrics. Use `wandb.Run.log()({'train_accuracy': 0.9, 'batch': 200})` in one step and `wandb.Run.log()({'val_accuracy': 0.8, 'epoch': 4})` in another. In the UI, set the desired value as the x-axis for each chart. To set a default x-axis for a specific index, use [Run.define_metric()]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ko" >}}). For the example provided, use the following code:

```python
import wandb

with wandb.init() as run:
   run.define_metric("batch")
   run.define_metric("epoch")

   run.define_metric("train_accuracy", step_metric="batch")
   run.define_metric("val_accuracy", step_metric="epoch")
```