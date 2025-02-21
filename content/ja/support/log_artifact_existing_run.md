---
title: How do I log an artifact to an existing run?
menu:
  support:
    identifier: ja-support-log_artifact_existing_run
tags:
- artifacts
toc_hide: true
type: docs
---

場合によっては、以前にログされた run の出力としてアーティファクトをマークする必要があります。この場合、古い run を再初期化して、新しいアーティファクトを以下のようにログしてください：

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```