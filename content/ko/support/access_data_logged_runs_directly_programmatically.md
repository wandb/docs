---
menu:
  support:
    identifier: ko-support-access_data_logged_runs_directly_programmatically
tags:
- experiments
title: How can I access the data logged to my runs directly and programmatically?
toc_hide: true
type: docs
---

The history object tracks metrics logged with `wandb.log`. Access the history object using the API:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```