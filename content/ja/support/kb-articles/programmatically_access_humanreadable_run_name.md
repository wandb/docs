---
menu:
  support:
    identifier: ja-support-kb-articles-programmatically_access_humanreadable_run_name
support:
- experiments
title: How do I programmatically access the human-readable run name?
toc_hide: true
type: docs
url: /support/:filename
---

The `.name` attribute of a [`wandb.Run`]({{< relref path="/ref/python/sdk/classes/run" lang="ja" >}}) is accessible as follows:

```python
import wandb

with wandb.init() as run:
   run_name = run.name
   print(f"The human-readable run name is: {run_name}")
```