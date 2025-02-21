---
title: How can I access the data logged to my runs directly and programmatically?
menu:
  support:
    identifier: ja-support-access_data_logged_runs_directly_programmatically
tags:
- experiments
toc_hide: true
type: docs
---

history オブジェクトは、`wandb.log` でログされたメトリクスを追跡します。API を使用して history オブジェクトにアクセスします。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```
