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

以前に ログ に記録された run の出力を artifact としてマークする必要がある場合があります。この場合、古い run を再度初期化し、次のように新しい Artifacts を ログ に記録します。

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```
