---
title: "How can I access the data logged to my runs directly and programmatically?"
tags:
   - experiments
---

The history object is used to track metrics logged by `wandb.log`. Using [our API](../guides/track/public-api-guide.md), you can access the history object via `run.history()`.

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```