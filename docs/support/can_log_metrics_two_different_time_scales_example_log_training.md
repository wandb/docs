---
title: "Can I log metrics on two different time scales?"
displayed_sidebar: support
tags:
   - None
---
For example, I want to log training accuracy per batch and validation accuracy per epoch.

Yes, log indices like `batch` and `epoch` alongside your metrics. Use `wandb.log({'train_accuracy': 0.9, 'batch': 200})` in one step and `wandb.log({'val_accuracy': 0.8, 'epoch': 4})` in another. In the UI, set the desired value as the x-axis for each chart. To set a default x-axis for a specific index, use [Run.define_metric()](../ref/python/run.md#define_metric). For the example provided, use the following code:

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```