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

`.name` 属性は、[`wandb.Run`]({{< relref path="/ref/python/run.md" lang="ja" >}}) で以下のように アクセス できます:

```python
import wandb

wandb.init()
run_name = wandb.run.name
```