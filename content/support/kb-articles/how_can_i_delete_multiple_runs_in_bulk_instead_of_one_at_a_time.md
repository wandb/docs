---
url: /support/:filename
title: How can I delete multiple runs in bulk instead of one at a time?
toc_hide: true
type: docs
support:
  - projects
  - runs
translationKey: how_can_i_delete_multiple_runs_in_bulk_instead_of_one_at_a_time
---
Use the [public API]({{< relref "/ref/python/public-api/api.md" >}}) to delete multiple runs in a single operation:

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()
```
