---
title: ラン名を run ID に設定するだけでいいのですか？
menu:
  support:
    identifier: ja-support-kb-articles-just_set_run_name_run_id
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

はい。 run ID で run 名を上書きするには、次のコードスニペットを使用します:

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```