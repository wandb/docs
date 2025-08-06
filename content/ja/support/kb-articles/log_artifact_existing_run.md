---
menu:
  support:
    identifier: ja-support-kb-articles-log_artifact_existing_run
support:
- artifacts
title: How do I log an artifact to an existing run?
toc_hide: true
type: docs
url: /support/:filename
---

Occasionally, it is necessary to mark an artifact as the output of a previously logged run. In this case, reinitialize the old run and log new artifacts as follows:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```