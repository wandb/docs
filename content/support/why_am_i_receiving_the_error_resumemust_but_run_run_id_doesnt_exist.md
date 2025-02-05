---
title: Why am I receiving the error "resume='must' but run (<run_id>) doesn't exist"?
toc_hide: true
type: docs
tags:
  - resuming
  - runs
---

This error arises when the run is not found under a project/entity. Ensure you are properly logged in and set `project` and `entity` in:

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

We recommend running `wandb login --relogin` to verify authentication.