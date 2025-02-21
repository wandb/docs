---
title: Can I just set the run name to the run ID?
menu:
  support:
    identifier: ja-support-just_set_run_name_run_id
tags:
- experiments
toc_hide: true
type: docs
---

はい。 run 名を run ID で上書きするには、次のコードスニペットを使用します。

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```
