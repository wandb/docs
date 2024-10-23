---
title: "How do I programmatically access the human-readable run name?"
displayed_sidebar: support
tags:
   - experiments
---
The `.name` attribute of a [`wandb.Run`](../ref/python/run.md) is accessible as follows:

```python
import wandb

wandb.init()
run_name = wandb.run.name
```