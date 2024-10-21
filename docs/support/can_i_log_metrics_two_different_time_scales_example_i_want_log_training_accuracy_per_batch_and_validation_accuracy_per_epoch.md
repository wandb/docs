---
title: "Can I log metrics on two different time scales? (For example, I want to log training accuracy per batch and validation accuracy per epoch.)"
tags: []
---

### Can I log metrics on two different time scales? (For example, I want to log training accuracy per batch and validation accuracy per epoch.)
Yes, you can do this by logging your indices (e.g. `batch` and `epoch`) whenever you log your other metrics. So in one step you could call `wandb.log({'train_accuracy': 0.9, 'batch': 200})` and in another step call `wandb.log({'val_accuracy': 0.8, 'epoch': 4})`. Then, in the UI, you can set the appropriate value as the x-axis for each chart. If you want to set the default x-axis of a particular index you can do so using by using [Run.define_metric()](../../ref/python/run.md#define_metric). In our above example we could do the following:

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```