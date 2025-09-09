---
title: プログラムから人間が読める run 名にアクセスするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-programmatically_access_humanreadable_run_name
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

[`wandb.Run`]({{< relref path="/ref/python/sdk/classes/run" lang="ja" >}}) の `.name` 属性は、次のように アクセス できます:

```python
import wandb

with wandb.init() as run:
   run_name = run.name
   print(f"The human-readable run name is: {run_name}")
```