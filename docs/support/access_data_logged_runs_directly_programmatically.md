---
title: "How can I access the data logged to my runs directly and programmatically?"
displayed_sidebar: support
tags:
   - experiments
---
The history object tracks metrics logged with `wandb.log`. Access the history object using the API:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```