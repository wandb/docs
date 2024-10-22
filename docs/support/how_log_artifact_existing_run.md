---
title: "How do I log an artifact to an existing run?"
tags:
   - artifacts
---

Occasionally, you may want to mark an artifact as the output of a previously logged run. In that scenario, you can [reinitialize the old run](../guides/runs/resuming.md) and log new artifacts to it as follows:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```