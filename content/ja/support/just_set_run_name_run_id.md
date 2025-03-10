---
menu:
  support:
    identifier: ja-support-just_set_run_name_run_id
tags:
- experiments
title: Can I just set the run name to the run ID?
toc_hide: true
type: docs
---

Yes. To overwrite the run name with the run ID, use the following code snippet:

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```