---
url: /support/:filename
title: "What if I want to log some metrics on batches and some metrics only on epochs?"
toc_hide: true
type: docs
support:
   - experiments
   - metrics
---
To log specific metrics in each batch and standardize plots, log the desired x-axis values alongside the metrics. In the custom plots, click edit and select a custom x-axis.

```python
import wandb

with wandb.init() as run:
    run.log({"batch": batch_idx, "loss": 0.3})
    run.log({"epoch": epoch, "val_acc": 0.94})
```