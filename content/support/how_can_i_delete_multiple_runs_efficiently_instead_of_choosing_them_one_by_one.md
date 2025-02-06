---
title: How can I delete multiple runs in bulk instead of one at a time?
toc_hide: true
type: docs
tags:
  - projects
  - runs
---

Use the Public API to delete multiple runs efficiently:

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()
```
