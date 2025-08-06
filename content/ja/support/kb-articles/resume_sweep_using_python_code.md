---
title: Python コードで sweep を再開するにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
- パイソン
---

sweep を再開するには、`sweep_id` を `wandb.agent()` 関数に渡します。

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # ここにトレーニングコードを記述
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```