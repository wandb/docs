---
title: "What if I want to log some metrics on batches and some metrics only on epochs?"
tags:
   - experiments
---

If you'd like to log certain metrics in every batch and standardize plots, you can log x axis values that you want to plot with your metrics. Then in the custom plots, click edit and select a custom x-axis.

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```