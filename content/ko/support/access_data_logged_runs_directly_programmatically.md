---
title: How can I access the data logged to my runs directly and programmatically?
menu:
  support:
    identifier: ko-support-access_data_logged_runs_directly_programmatically
tags:
- experiments
toc_hide: true
type: docs
---

history 오브젝트는 `wandb.log` 로깅된 메트릭을 추적합니다. API를 사용하여 history 오브젝트에 액세스하세요.

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```
