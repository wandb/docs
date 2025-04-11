---
title: Python コードを使って sweep を再開するにはどうすればいいですか？
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

To resume a sweep, pass the `sweep_id` to the `wandb.agent()` function.

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # トレーニングコードはこちら
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```