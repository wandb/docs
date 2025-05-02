---
title: How can I delete multiple runs in bulk instead of one at a time?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_delete_multiple_runs_in_bulk_instead_of_one_at_a_time
support:
- projects
- runs
toc_hide: true
type: docs
url: /ko/support/:filename
---

[public API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})를 사용하여 단일 작업에서 여러 개의 runs을 삭제합니다.

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()
```
