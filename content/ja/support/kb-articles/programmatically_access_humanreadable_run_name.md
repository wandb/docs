---
title: プログラムから人間が読める run 名にアクセスするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

`.name` 属性は、[`wandb.Run`]({{< relref "/ref/python/sdk/classes/run" >}}) から以下のようにアクセスできます。

```python
import wandb

with wandb.init() as run:
   run_name = run.name
   # 人間が読みやすい run の名前を出力します
   print(f"The human-readable run name is: {run_name}")
```