---
title: How do I use the resume parameter when resuming a run in W&B?
menu:
  support:
    identifier: ja-support-resume_parameter
tags:
- resuming
toc_hide: true
type: docs
---

`resume` パラメータを W&B で使用するには、`entity`、`project`、および `id` を指定して `wandb.init()` の `resume` 引数を設定します。`resume` 引数は `"must"` または `"allow"` の値を受け入れます。

  ```python
  run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
  ```