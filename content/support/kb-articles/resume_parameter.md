---
url: /support/:filename
title: How do I use the resume parameter when resuming a run in W&B?  
toc_hide: true
type: docs
support:
- resuming
---
To use the `resume` parameter in W&B , set the `resume` argument in `wandb.init()` with `entity`, `project`, and `id` specified. The `resume` argument accepts values of `"must"` or `"allow"`. 

  ```python
  run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
  ```