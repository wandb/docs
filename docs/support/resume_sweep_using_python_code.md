---
title: How can I resume a sweep using Python code?
displayed_sidebar: support
tags:
- sweeps
---
To resume a sweep, pass the `sweep_id` to the `wandb.agent()` function. Here's an example:

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # Your training code here
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```