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

W&B で `resume` パラメータを使用するには、`wandb.init()` で `resume` 引数に `entity` 、 `project` 、および `id` を指定して設定します。 `resume` 引数は、 `"must"` または `"allow"` の 値を受け入れます。

```python
run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
```
