---
title: How do I programmatically access the human-readable run name?
menu:
  support:
    identifier: ja-support-kb-articles-programmatically_access_humanreadable_run_name
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

[`wandb.Run`]({{< relref path="/ref/python/run.md" lang="ja" >}}) の `.name` 属性には、以下のようにアクセスできます。

```python
import wandb

wandb.init()
run_name = wandb.run.name
```
