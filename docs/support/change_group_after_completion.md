---
title: Is it possible to change the group assigned to a run after completion?
displayed_sidebar: support
tags:
- runs 
---
You can change the group assigned to a completed run using the API. This feature does not appear in the web UI. Use the following code to update the group:

```python
import wandb

api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"
run.update()
```