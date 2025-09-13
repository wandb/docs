---
title: W&B で run を再開する際に resume パラメータはどう使えばよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-resume_parameter
support:
- 再開
toc_hide: true
type: docs
url: /support/:filename
---

W&B で `resume` パラメータを使用するには、`entity`、`project`、`id` を指定して `wandb.init()` の `resume` 引数を設定します。`resume` 引数には "must" または "allow" の値を指定できます。

```python
run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
```