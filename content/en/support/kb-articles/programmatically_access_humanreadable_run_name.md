---
url: /support/:filename
title: "How do I programmatically access the human-readable run name?"
toc_hide: true
type: docs
support:
   - experiments
---
The `.name` attribute of a [`wandb.Run`]({{< relref "/ref/python/sdk/experiments/run" >}}) is accessible as follows:

```python
import wandb

with wandb.init() as run:
   run_name = run.name
   print(f"The human-readable run name is: {run_name}")
```