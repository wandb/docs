---
title: "How do I find an artifact from the best run in a sweep?"
tags:
   - artifacts
---

You can use the following code to retrieve the artifacts associated with the best performing run in a sweep:

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```