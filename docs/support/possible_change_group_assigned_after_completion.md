---
title: Is it possible to change the group assigned to a run after completion?
displayed_sidebar: support
tags:
- workspaces 
---
Yes, you can change the group assigned to a run after its completion using the API. This is not available through the web UI. Use the following code to update the group:

```python
import wandb

api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"
run.update()
```

You can request to be notified about updates on a feature that supports this in the UI.
    