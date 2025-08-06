---
title: 한 번에 하나씩이 아니라 여러 run을 한꺼번에 삭제할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_delete_multiple_runs_in_bulk_instead_of_one_at_a_time
support:
- 프로젝트
- run
toc_hide: true
type: docs
url: /support/:filename
---

[public API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})를 사용하여 한 번에 여러 Runs를 삭제할 수 있습니다:

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()
```