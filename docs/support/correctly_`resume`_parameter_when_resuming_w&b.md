---
title: How do I correctly use the `resume` parameter when resuming a run in W&B?  
displayed_sidebar: support
tags:
- resuming
---
To use the `resume` parameter in W&B correctly:

- Use `wandb. init()`'s `resume` argument with `entity,` `project,` and `id` specified in `wandb.init()`. The `resume` argument can be set to `"must"`or `"allow"`. 
- Example: 
  ```python
  run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
  ```