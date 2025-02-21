---
title: How do I programmatically access the human-readable run name?
menu:
  support:
    identifier: ja-support-programmatically_access_humanreadable_run_name
tags:
- experiments
toc_hide: true
type: docs
---

[`wandb.Run`]({{< relref path="/ref/python/run.md" lang="ja" >}}) の `.name` 属性には、次のようにアクセスできます。

```python
import wandb

wandb.init()
run_name = wandb.run.name
```
