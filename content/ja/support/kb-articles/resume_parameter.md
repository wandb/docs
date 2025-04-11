---
title: W&B で run を再開する際に resume パラメータをどのように使用しますか？
menu:
  support:
    identifier: ja-support-kb-articles-resume_parameter
support:
- resuming
toc_hide: true
type: docs
url: /support/:filename
---

W&B で `resume` パラメータを使用するには、`resume` 引数を `entity`、`project`、および `id` を指定して `wandb.init()` に設定します。`resume` 引数は `"must"` または `"allow"` の値を受け付けます。

  ```python
  run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
  ```