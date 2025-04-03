---
title: How can I resume a sweep using Python code?
menu:
  support:
    identifier: ja-support-kb-articles-resume_sweep_using_python_code
support:
- sweeps
- python
toc_hide: true
type: docs
url: /support/:filename
---

sweep を再開するには、 `sweep_id` を `wandb.agent()` 関数に渡します。

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # Training code here (ここにトレーニング コードを記述)
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```
