---
title: "How do I programmatically access the human-readable run name?"
tags:
   - experiments
---

It's available as the `.name` attribute of a [`wandb.Run`](../../ref/python/run.md).

```python
import wandb

wandb.init()
run_name = wandb.run.name
```