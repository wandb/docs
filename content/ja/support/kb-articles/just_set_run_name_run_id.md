---
title: 単に run 名を run ID に設定するだけでいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-just_set_run_name_run_id
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

はい。run の名前を run ID で上書きするには、次の コードスニペット を使用してください：

```python
import wandb

with wandb.init() as run:
   run.name = run.id
   run.save()
```