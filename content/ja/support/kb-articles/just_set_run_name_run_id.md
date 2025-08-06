---
title: run 名を run ID に設定してもいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-just_set_run_name_run_id
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

はい。run の名前を run ID で上書きするには、次のコードスニペットを使用してください。

```python
import wandb

with wandb.init() as run:
   # run の名前を run ID で上書き
   run.name = run.id
   run.save()
```