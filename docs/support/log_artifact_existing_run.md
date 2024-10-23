---
title: "How do I log an artifact to an existing run?"
displayed_sidebar: support
tags:
   - artifacts
---
Occasionally, it is necessary to mark an artifact as the output of a previously logged run. In this case, reinitialize the old run and log new artifacts as follows:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```