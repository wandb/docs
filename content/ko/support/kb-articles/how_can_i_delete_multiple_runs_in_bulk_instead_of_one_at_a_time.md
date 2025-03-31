---
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_delete_multiple_runs_in_bulk_instead_of_one_at_a_time
support:
- projects
- runs
title: How can I delete multiple runs in bulk instead of one at a time?
toc_hide: true
type: docs
url: /support/:filename
---

Use the [public API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}) to delete multiple runs in a single operation:

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()
```