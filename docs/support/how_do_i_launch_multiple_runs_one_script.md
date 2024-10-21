---
title: "How do I launch multiple runs from one script?"
tags:
   - experiments
---

Use `wandb.init` and `run.finish()` to log multiple Runs from one script:

1. `run = wandb.init(reinit=True)`: Use this setting to allow reinitializing runs
2. `run.finish()`: Use this at the end of your run to finish logging for that run

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

Alternatively you can use a python context manager which will automatically finish logging:

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```