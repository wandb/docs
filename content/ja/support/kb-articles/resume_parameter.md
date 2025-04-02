---
title: How do I use the resume parameter when resuming a run in W&B?
menu:
  support:
    identifier: ja-support-kb-articles-resume_parameter
support:
- resuming
toc_hide: true
type: docs
url: /support/:filename
---

W&B で `resume` パラメータを使用するには、`wandb.init()` で `resume` 引数を設定し、`entity`、`project`、および `id` を指定します。`resume` 引数は `"must"` または `"allow"` の 値を受け入れます。

```python
run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
```
