---
title: ラン名を run ID に設定するだけでいいのですか？
menu:
  support:
    identifier: ja-support-kb-articles-just_set_run_name_run_id
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: just_set_run_name_run_id
---
はい。 run ID で run 名を上書きするには、次のコードスニペットを使用します:

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```