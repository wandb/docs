---
title: How do I fix the error `resume='must' but run (<run_id>) doesn't exist`?
toc_hide: true
type: docs
tags:
  - resuming
  - runs
---

If you encounter the error `resume='must' but run (<run_id>) doesn't exist`, the run you are attempting to resume does not exist within the project or entity. Ensure that you are logged in to the correct instance and that the project and entity are set:

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

We recommend running `wandb login --relogin` to verify authentication.