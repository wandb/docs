---
title: "How do I launch multiple runs from one script?"
displayed_sidebar: support
tags:
   - experiments
---
Use `wandb.init` and `run.finish()` to log multiple runs within a single script:

1. Use `run = wandb.init(reinit=True)` to allow reinitialization of runs.
2. Call `run.finish()` at the end of each run to complete logging.

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

Alternatively, utilize a Python context manager to automatically finish logging:

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```