---
title: Can I just set the run name to the run ID?
menu:
  support:
    identifier: ja-support-kb-articles-just_set_run_name_run_id
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

はい。run名を Run ID で上書きするには、次のコードスニペットを使用します。

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```
