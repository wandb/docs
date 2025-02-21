---
title: How can I delete multiple runs in bulk instead of one at a time?
menu:
  support:
    identifier: ja-support-how_can_i_delete_multiple_runs_in_bulk_instead_of_one_at_a_time
tags:
- projects
- runs
toc_hide: true
type: docs
---

[public API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用して、一度に複数の run を削除します:

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()
```