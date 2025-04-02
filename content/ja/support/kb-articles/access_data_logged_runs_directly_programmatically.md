---
title: How can I access the data logged to my runs directly and programmatically?
menu:
  support:
    identifier: ja-support-kb-articles-access_data_logged_runs_directly_programmatically
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

history オブジェクトは、`wandb.log` でログされたメトリクスを追跡します。 history オブジェクトには、API を使用してアクセスします。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```
