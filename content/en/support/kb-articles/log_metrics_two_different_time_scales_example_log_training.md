---
url: /support/:filename
title: "Can I log metrics on two different time scales?"
toc_hide: true
type: docs
support:
   - experiments
   - metrics
---
For example, I want to log training accuracy per batch and validation accuracy per epoch.

Yes, log indices like `batch` and `epoch` alongside your metrics. Use `wandb.Run.log()({'train_accuracy': 0.9, 'batch': 200})` in one step and `wandb.Run.log()({'val_accuracy': 0.8, 'epoch': 4})` in another. In the UI, set the desired value as the x-axis for each chart. To set a default x-axis for a specific index, use [Run.define_metric()]({{< relref "/ref/python/sdk/experiments/run#define_metric" >}}). For the example provided, use the following code:

```python
import wandb

with wandb.init() as run:
   run.define_metric("batch")
   run.define_metric("epoch")

   run.define_metric("train_accuracy", step_metric="batch")
   run.define_metric("val_accuracy", step_metric="epoch")
```