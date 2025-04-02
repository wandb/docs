---
title: Can I just set the run name to the run ID?
menu:
  support:
    identifier: ko-support-kb-articles-just_set_run_name_run_id
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

예: run 이름을 run ID로 덮어쓰려면 다음 코드 조각을 사용하세요.

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```
