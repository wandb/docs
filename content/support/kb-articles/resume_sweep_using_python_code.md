---
url: /support/:filename
title: How can I resume a sweep using Python code?
toc_hide: true
type: docs
support:
- sweeps
- python
---
To resume a sweep, pass the `sweep_id` to the `wandb.agent()` function. 

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # Training code here
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```