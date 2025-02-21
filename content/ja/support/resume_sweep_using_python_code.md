---
title: How can I resume a sweep using Python code?
menu:
  support:
    identifier: ja-support-resume_sweep_using_python_code
tags:
- sweeps
- python
toc_hide: true
type: docs
---

sweep を再開するには、`sweep_id` を `wandb.agent()` 関数に渡します。

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # トレーニング コードをここに記述
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```
