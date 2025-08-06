---
title: 내 run 에 기록된 데이터에 직접 또는 프로그래밍적으로 접근하려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-access_data_logged_runs_directly_programmatically
support:
- Experiments
toc_hide: true
type: docs
url: /support/:filename
---

history 오브젝트는 `wandb.log` 로 기록된 메트릭을 추적합니다. API를 사용하여 history 오브젝트에 접근할 수 있습니다:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```