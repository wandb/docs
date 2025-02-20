---
menu:
  support:
    identifier: ko-support-resume_sweep_using_python_code
tags:
- sweeps
- python
title: How can I resume a sweep using Python code?
toc_hide: true
type: docs
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