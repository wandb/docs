---
title: run 名を run ID に設定するだけでもいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

はい。run 名を run ID で上書きするには、次のコードスニペットを使用してください。

```python
import wandb

with wandb.init() as run:
   # run 名を run ID で上書き
   run.name = run.id
   run.save()
```