---
title: "How do I launch multiple runs from one script?"
toc_hide: true
type: docs
tags:
   - experiments
---
Finish previous runs before starting new runs to log multiple runs within
a single script.

The recommended way to do this is by using `wandb.init()` as a context manager
because this finishes the run and marks it as failed if your script raises an
exception:

```python
import wandb

for x in range(10):
    with wandb.init() as run:
        for y in range(100):
            run.log({"metric": x + y})
```

You can also call `run.finish()` explicitly:

```python
import wandb

for x in range(10):
    run = wandb.init()

    try:
        for y in range(100):
            run.log({"metric": x + y})

    except Exception:
        run.finish(exit_code=1)
        raise

    finally:
        run.finish()
```

## Multiple active runs

Starting with wandb 0.19.9, you can set the `reinit` setting to `"allow"` to
create multiple simultaneously active runs. The recommended way to do this
is by calling `wandb.setup()` after importing `wandb`.


```python
import wandb

wandb.setup(wandb.Settings(reinit="allow"))

with wandb.init() as tracking_run:
    for x in range(10):
        with wandb.init() as run:
            for y in range(100):
                run.log({"x_plus_y": x + y})

            tracking_run.log({"x": x})
```
