---
title: W&B で run を再開する際に resume パラメータはどのように使いますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 再開
---

W&B で `resume` パラメータを使うには、`wandb.init()` で `entity`、`project`、`id` を指定し、`resume` 引数を設定します。`resume` 引数には `"must"` または `"allow"` の値を指定できます。

  ```python
  run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
  ```