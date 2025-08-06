---
title: プログラムから人が読める run 名にアクセスするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-programmatically_access_humanreadable_run_name
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

`.name` 属性は、[`wandb.Run`]({{< relref path="/ref/python/sdk/classes/run" lang="ja" >}}) から次のようにアクセスできます。

```python
import wandb

with wandb.init() as run:
   run_name = run.name
   print(f"人間が読める run の名前は: {run_name}")
```